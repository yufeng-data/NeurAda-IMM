"""NeurAda-IMM 完整仿真实验（支持真实数据测试）

说明：
- 原版单文件 `nuScene_main.py` 已按“代码块/模块”拆分到 `code/neurada_imm/` 包中。
- 该文件现在作为入口脚本（main runner），负责串联训练/测试与绘图。

模块划分：
- `neurada_imm/config.py`：全局配置（DT/噪声/训练轮数/nuScenes路径等）
- `neurada_imm/data_synth.py`：合成数据生成 + 上下文 + 加噪
- `neurada_imm/data_nuscenes.py`：nuScenes真实轨迹提取与重采样
- `neurada_imm/datasets.py`：PyTorch Dataset
- `neurada_imm/models.py`：神经网络组件
- `neurada_imm/filters.py`：NNKFWrapper / TraditionalIMM / NeurAdaIMM
- `neurada_imm/train.py`：预训练与微调
- `neurada_imm/metrics.py`：评估指标
- `neurada_imm/plotting.py`：绘图
"""

import numpy as np

# -------------------------- 可选依赖：nuScenes --------------------------
try:
    from nuscenes.nuscenes import NuScenes
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False
    print("警告：未安装nuscenes-devkit，无法加载真实数据。")

from neurada_imm.config import (
    DT,
    T_STEPS,
    OBS_SIGMA,
    EPOCHS_PRETRAIN,
    EPOCHS_FINETUNE,
    L_MODELS,
    K_ACTIVE,
    CONTEXT_DIM,
    NUSCENES_DATAROOT,
    NUSCENES_VERSION,
    PROCESS_NOISE,
    QV_SCALE,
)
from neurada_imm.data_synth import (
    generate_trajectory_with_modes,
    generate_context,
    add_measurement_noise,
)
from neurada_imm.data_nuscenes import load_nuscenes_trajectory
from neurada_imm.filters import TraditionalIMM, NeurAdaIMM
from neurada_imm.train import pretrain_model_bank, finetune_adaptive_nets
from neurada_imm.metrics import compute_motve, compute_position_rmse, compute_mas, print_comparison_table
from neurada_imm.plotting import plot_tracking_comparison, plot_nuscenes_tracking
from neurada_imm.checkpoint import save_checkpoint


def _make_Q(q_base: np.ndarray, qv_scale: float) -> np.ndarray:
    Q = q_base.astype(np.float32).copy()
    Q[2, 2] *= float(qv_scale)
    Q[3, 3] *= float(qv_scale)
    return Q


def test_on_nuscenes(model_bank, selector, trans_net, weight_net):
    if not NUSCENES_AVAILABLE:
        print("未安装nuScenes-devkit，跳过真实数据测试。")
        return

    print("\n" + "=" * 60)
    print("开始真实数据测试（nuScenes）")
    print("=" * 60)

    try:
        nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)
    except Exception as e:
        print(f"无法加载nuScenes数据：{e}，请检查路径/版本。")
        return

    try:
        true_states, timestamps = load_nuscenes_trajectory(
            nusc,
            scene_idx=0,
            category_name='human.pedestrian.adult',
            dt=DT,
        )
    except Exception as e:
        print(f"轨迹提取失败：{e}")
        return

    T = len(true_states)
    print(f"轨迹长度：{T} 步，时间范围：{timestamps[0]:.2f} ~ {timestamps[-1]:.2f} s")

    noisy_meas = add_measurement_noise(true_states, sigma=OBS_SIGMA)

    # 简化上下文
    cat_onehot = np.array([1.0, 0.0], dtype=np.float32)
    density = 0.5
    intersection = 0.0
    context = np.zeros((T, CONTEXT_DIM), dtype=np.float32)
    context[:, :2] = cat_onehot
    context[:, 2] = density
    context[:, 3] = intersection
    context[:, 4] = 0.0

    print("\n运行传统IMM...")
    Q_cv = np.diag([0.1, 0.1, 0.5, 0.5, 0.01, 0.01]).astype(np.float32)
    Q_ctrv = np.diag([0.1, 0.1, 1.0, 1.0, 0.1, 0.1]).astype(np.float32)
    R = (np.eye(2) * OBS_SIGMA ** 2).astype(np.float32)
    trans_matrix = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=np.float32)

    imm_trad = TraditionalIMM(DT, Q_cv, Q_ctrv, R, trans_matrix)
    imm_trad.initialize(noisy_meas[0, :2])
    trad_est = [imm_trad.x_combined.copy()]
    for t in range(1, T):
        trad_est.append(imm_trad.step(noisy_meas[t, :2]).copy())
    trad_est = np.asarray(trad_est, dtype=np.float32)

    print("运行NeurAda-IMM...")
    model_bank_cpu = [m.cpu() for m in model_bank]
    selector_cpu = selector.cpu()
    trans_net_cpu = trans_net.cpu()
    weight_net_cpu = weight_net.cpu()

    neur_imm = NeurAdaIMM(DT, model_bank_cpu, selector_cpu, trans_net_cpu, weight_net_cpu,
                          Q=_make_Q(PROCESS_NOISE, QV_SCALE),
                          R=R, K=K_ACTIVE, context_dim=CONTEXT_DIM)
    neur_imm.initialize(noisy_meas[0, :2], context[0])
    neur_est = [neur_imm.x_combined.copy()]
    for t in range(1, T):
        context[t, 4] = float(np.linalg.norm(neur_imm.x_combined[2:4]))
        neur_est.append(neur_imm.step(noisy_meas[t, :2], context[t]).copy())
    neur_est = np.asarray(neur_est, dtype=np.float32)

    motve_trad = compute_motve(trad_est, true_states)
    motve_neur = compute_motve(neur_est, true_states)
    pos_rmse_trad = compute_position_rmse(trad_est, true_states)
    pos_rmse_neur = compute_position_rmse(neur_est, true_states)

    print("\n真实数据测试结果：")
    print_comparison_table({
        'Traditional IMM': {'MOTVE': motve_trad, 'Position RMSE': pos_rmse_trad},
        'NeurAda-IMM': {'MOTVE': motve_neur, 'Position RMSE': pos_rmse_neur},
    })

    plot_nuscenes_tracking(true_states, noisy_meas, trad_est, neur_est)


def main():
    print("=" * 60)
    print("NeurAda-IMM 完整仿真实验 (模块化版本)")
    print("=" * 60)

    print("\n生成训练轨迹...")
    train_trajs = []
    train_ctxs = []
    train_modes_list = []
    for _ in range(20):
        traj, modes = generate_trajectory_with_modes(num_steps=T_STEPS, density_level='medium')
        ctx = generate_context(modes, density_value=0.5)
        train_trajs.append(traj)
        train_ctxs.append(ctx)
        train_modes_list.append(modes)

    print("\n预训练模型库...")
    model_bank = pretrain_model_bank(train_trajs, L=L_MODELS, epochs=EPOCHS_PRETRAIN)

    print("\n端到端微调自适应网络...")
    from torch.utils.data import DataLoader
    from neurada_imm.datasets import FinetuneDataset

    finetune_dataset = FinetuneDataset(train_trajs, train_ctxs, train_modes_list)
    finetune_loader = DataLoader(finetune_dataset, batch_size=1, shuffle=True)
    selector, trans_net, weight_net = finetune_adaptive_nets(model_bank, finetune_loader, epochs=EPOCHS_FINETUNE)

    # 训练完成后保存模型
    save_checkpoint(model_bank, selector, trans_net, weight_net, path='checkpoint.pth')

    # 合成数据对比测试
    print("\n" + "=" * 60)
    print("合成数据对比测试（NeurAda-IMM vs 传统IMM）")
    print("=" * 60)
    traj_test, modes_test = generate_trajectory_with_modes(num_steps=T_STEPS, density_level='medium')
    ctx_test = generate_context(modes_test, density_value=0.5)
    noisy_meas_test = add_measurement_noise(traj_test, sigma=OBS_SIGMA)

    Q_cv = np.diag([0.1, 0.1, 0.5, 0.5, 0.01, 0.01]).astype(np.float32)
    Q_ctrv = np.diag([0.1, 0.1, 1.0, 1.0, 0.1, 0.1]).astype(np.float32)
    R = (np.eye(2) * OBS_SIGMA ** 2).astype(np.float32)
    trans_matrix = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=np.float32)

    imm_trad = TraditionalIMM(DT, Q_cv, Q_ctrv, R, trans_matrix)
    imm_trad.initialize(noisy_meas_test[0, :2])
    trad_est_synth = [imm_trad.x_combined.copy()]
    for t in range(1, T_STEPS):
        trad_est_synth.append(imm_trad.step(noisy_meas_test[t, :2]).copy())
    trad_est_synth = np.asarray(trad_est_synth, dtype=np.float32)

    import torch
    model_bank_cpu = [m.cpu() for m in model_bank]
    selector_cpu = selector.cpu()
    trans_net_cpu = trans_net.cpu()
    weight_net_cpu = weight_net.cpu()

    neur_imm_synth = NeurAdaIMM(DT, model_bank_cpu, selector_cpu, trans_net_cpu, weight_net_cpu,
                                Q=_make_Q(PROCESS_NOISE, QV_SCALE),
                                R=R, K=K_ACTIVE, context_dim=CONTEXT_DIM)
    neur_imm_synth.initialize(noisy_meas_test[0, :2], ctx_test[0])
    neur_est_synth = [neur_imm_synth.x_combined.copy()]
    for t in range(1, T_STEPS):
        ctx_test[t, 4] = float(np.linalg.norm(neur_imm_synth.x_combined[2:4]))
        neur_est_synth.append(neur_imm_synth.step(noisy_meas_test[t, :2], ctx_test[t]).copy())
    neur_est_synth = np.asarray(neur_est_synth, dtype=np.float32)

    print_comparison_table({
        'Traditional IMM': {
            'MOTVE': compute_motve(trad_est_synth, traj_test),
            'Position RMSE': compute_position_rmse(trad_est_synth, traj_test),
        },
        'NeurAda-IMM': {
            'MOTVE': compute_motve(neur_est_synth, traj_test),
            'Position RMSE': compute_position_rmse(neur_est_synth, traj_test),
        },
    })

    plot_tracking_comparison(traj_test, noisy_meas_test, trad_est_synth, neur_est_synth,
                             out_path='tracking_comparison.png')

    # 真实数据测试
    test_on_nuscenes(model_bank, selector, trans_net, weight_net)

    print("\n实验完成！")



if __name__ == '__main__':
    main()