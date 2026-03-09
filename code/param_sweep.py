"""小型参数扫描脚本：对 NeurAda-IMM 的关键滤波参数做网格搜索，并输出 MOTVE/PosRMSE 表格。

用法（PowerShell示例）：
  python .\param_sweep.py

说明：
- 该脚本会复用现有训练流程（合成数据：预训练模型库 + 微调自适应网络），然后在固定测试轨迹上对不同参数组合评估。
- 为避免训练波动影响对比：训练与测试都固定随机种子；同一轮 sweep 使用同一组训练结果。
- 扫描参数：
    1) Q 的速度噪声倍率 qv_scale（作用于 PROCESS_NOISE 的 vx/vy）
    2) 融合权重 EMA 平滑系数 ema_lambda（NeurAdaIMM 内部参数）

注意：
- 若你后续想把扫描扩展到 R/OBS_SIGMA、或对 nuScenes 做 sweep，可以在 TODO 位置继续加。
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np

from neurada_imm.config import (
    DT,
    T_STEPS,
    OBS_SIGMA,
    EPOCHS_PRETRAIN,
    EPOCHS_FINETUNE,
    L_MODELS,
    K_ACTIVE,
    CONTEXT_DIM,
    PROCESS_NOISE,
)
from neurada_imm.data_synth import generate_trajectory_with_modes, generate_context, add_measurement_noise
from neurada_imm.datasets import FinetuneDataset
from neurada_imm.filters import TraditionalIMM, NeurAdaIMM
from neurada_imm.metrics import compute_motve, compute_position_rmse
from neurada_imm.train import pretrain_model_bank, finetune_adaptive_nets


def _set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _make_Q_from_process_noise(process_noise: np.ndarray, qv_scale: float) -> np.ndarray:
    Q = process_noise.astype(np.float32).copy()
    Q[2, 2] *= float(qv_scale)
    Q[3, 3] *= float(qv_scale)
    return Q


@dataclass
class SweepResult:
    qv_scale: float
    ema_lambda: float
    motve_trad: float
    rmse_trad: float
    motve_neur: float
    rmse_neur: float
    time_ms: float


def _eval_once(
    *,
    model_bank,
    selector,
    trans_net,
    weight_net,
    test_traj: np.ndarray,
    test_ctx: np.ndarray,
    qv_scale: float,
    ema_lambda: float,
) -> SweepResult:
    noisy_meas = add_measurement_noise(test_traj, sigma=OBS_SIGMA)

    # Traditional IMM
    Q_cv = np.diag([0.1, 0.1, 0.5, 0.5, 0.01, 0.01]).astype(np.float32)
    Q_ctrv = np.diag([0.1, 0.1, 1.0, 1.0, 0.1, 0.1]).astype(np.float32)
    R = (np.eye(2) * OBS_SIGMA**2).astype(np.float32)
    trans_matrix = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=np.float32)

    imm_trad = TraditionalIMM(DT, Q_cv, Q_ctrv, R, trans_matrix)
    imm_trad.initialize(noisy_meas[0, :2])
    trad_est = [imm_trad.x_combined.copy()]
    for t in range(1, len(noisy_meas)):
        trad_est.append(imm_trad.step(noisy_meas[t, :2]).copy())
    trad_est = np.asarray(trad_est, dtype=np.float32)

    # NeurAda-IMM
    model_bank_cpu = [m.cpu() for m in model_bank]
    selector_cpu = selector.cpu()
    trans_net_cpu = trans_net.cpu()
    weight_net_cpu = weight_net.cpu()

    Q = _make_Q_from_process_noise(PROCESS_NOISE, qv_scale=qv_scale)

    neur_imm = NeurAdaIMM(
        DT,
        model_bank_cpu,
        selector_cpu,
        trans_net_cpu,
        weight_net_cpu,
        Q=Q,
        R=R,
        K=K_ACTIVE,
        context_dim=CONTEXT_DIM,
    )

    # 覆盖EMA参数（不改类接口，直接写字段）
    if hasattr(neur_imm, "_ema_lambda"):
        neur_imm._ema_lambda = float(ema_lambda)

    neur_imm.initialize(noisy_meas[0, :2], test_ctx[0])

    t0 = time.perf_counter()
    neur_est = [neur_imm.x_combined.copy()]
    for t in range(1, len(noisy_meas)):
        neur_est.append(neur_imm.step(noisy_meas[t, :2], test_ctx[t]).copy())
    neur_est = np.asarray(neur_est, dtype=np.float32)
    t1 = time.perf_counter()

    motve_trad = compute_motve(trad_est, test_traj)
    rmse_trad = compute_position_rmse(trad_est, test_traj)
    motve_neur = compute_motve(neur_est, test_traj)
    rmse_neur = compute_position_rmse(neur_est, test_traj)

    return SweepResult(
        qv_scale=float(qv_scale),
        ema_lambda=float(ema_lambda),
        motve_trad=float(motve_trad),
        rmse_trad=float(rmse_trad),
        motve_neur=float(motve_neur),
        rmse_neur=float(rmse_neur),
        time_ms=float((t1 - t0) * 1000.0),
    )


def _print_table(results: list[SweepResult]):
    # sort: neurada PosRMSE then MOTVE
    results = sorted(results, key=lambda r: (r.rmse_neur, r.motve_neur))

    headers = [
        "qv_scale",
        "ema",
        "Trad_PosRMSE",
        "Trad_MOTVE",
        "Neur_PosRMSE",
        "Neur_MOTVE",
        "delta_RMSE(Neur-Trad)",
        "delta_MOTVE(Neur-Trad)",
        "Neur_time_ms",
    ]

    rows = []
    for r in results:
        rows.append(
            [
                f"{r.qv_scale:>7.2f}",
                f"{r.ema_lambda:>4.2f}",
                f"{r.rmse_trad:>12.4f}",
                f"{r.motve_trad:>10.4f}",
                f"{r.rmse_neur:>11.4f}",
                f"{r.motve_neur:>9.4f}",
                f"{(r.rmse_neur - r.rmse_trad):>18.4f}",
                f"{(r.motve_neur - r.motve_trad):>19.4f}",
                f"{r.time_ms:>12.2f}",
            ]
        )

    # pretty print (simple)
    col_w = [max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    def _fmt_line(cols):
        return " | ".join(c.ljust(col_w[i]) for i, c in enumerate(cols))

    sep = "-+-".join("-" * w for w in col_w)

    print("\n" + _fmt_line(headers))
    print(sep)
    for row in rows:
        print(_fmt_line(row))


def main():
    _set_seed(0)

    # 1) training data
    train_trajs = []
    train_ctxs = []
    train_modes_list = []
    for _ in range(20):
        traj, modes = generate_trajectory_with_modes(num_steps=T_STEPS, density_level='medium')
        ctx = generate_context(modes, density_value=0.5)
        train_trajs.append(traj)
        train_ctxs.append(ctx)
        train_modes_list.append(modes)

    # 2) train once
    print("Training model bank...")
    model_bank = pretrain_model_bank(train_trajs, L=L_MODELS, epochs=EPOCHS_PRETRAIN)

    print("Finetuning adaptive nets...")
    from torch.utils.data import DataLoader

    finetune_dataset = FinetuneDataset(train_trajs, train_ctxs, train_modes_list)
    finetune_loader = DataLoader(finetune_dataset, batch_size=1, shuffle=True)
    selector, trans_net, weight_net = finetune_adaptive_nets(model_bank, finetune_loader, epochs=EPOCHS_FINETUNE)

    # 3) fixed test trajectory for fair comparison
    _set_seed(123)
    test_traj, test_modes = generate_trajectory_with_modes(num_steps=T_STEPS, density_level='dense')
    test_ctx = generate_context(test_modes, density_value=0.8)

    # 4) grid
    qv_scales = [0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
    ema_lambdas = [0.80, 0.85, 0.90, 0.93]

    print("\nSweeping...")
    results: list[SweepResult] = []
    for qv_scale, ema in itertools.product(qv_scales, ema_lambdas):
        _set_seed(999)  # keep measurement noise comparable across configs
        r = _eval_once(
            model_bank=model_bank,
            selector=selector,
            trans_net=trans_net,
            weight_net=weight_net,
            test_traj=test_traj,
            test_ctx=test_ctx,
            qv_scale=qv_scale,
            ema_lambda=ema,
        )
        results.append(r)
        print(
            f"qv_scale={qv_scale:.2f}, ema={ema:.2f} | "
            f"Neur PosRMSE={r.rmse_neur:.4f}, MOTVE={r.motve_neur:.4f} | "
            f"Trad PosRMSE={r.rmse_trad:.4f}, MOTVE={r.motve_trad:.4f}"
        )

    _print_table(results)

    best = sorted(results, key=lambda x: (x.rmse_neur, x.motve_neur))[0]
    print(
        "\nBEST (by Neur PosRMSE then MOTVE): "
        f"qv_scale={best.qv_scale:.2f}, ema={best.ema_lambda:.2f} | "
        f"Neur PosRMSE={best.rmse_neur:.4f}, MOTVE={best.motve_neur:.4f}"
    )


if __name__ == "__main__":
    main()
