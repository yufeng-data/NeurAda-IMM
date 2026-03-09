import numpy as np


def compute_motve(est_states, true_states):
    v_est = np.linalg.norm(est_states[:, 2:4], axis=1)
    v_true = np.linalg.norm(true_states[:, 2:4], axis=1)
    return float(np.sqrt(np.mean((v_est - v_true) ** 2)))


def compute_position_rmse(est_states, true_states):
    return float(np.sqrt(np.mean(np.sum((est_states[:, :2] - true_states[:, :2]) ** 2, axis=1))))


def compute_mas(selected_history, true_modes, L: int):
    correct = 0
    total = len(selected_history)
    mode_groups = {
        0: list(range(L // 2)),
        1: list(range(L // 2, L)),
        2: list(range(L // 2, L)),
        3: [0],
    }
    for t, indices in enumerate(selected_history):
        if t >= len(true_modes):
            break
        mode = int(true_modes[t])
        if any(int(idx) in mode_groups.get(mode, []) for idx in indices):
            correct += 1
    return correct / max(total, 1)


def compute_velocity_rmse(est_states, true_states):
    """各速度分量 (vx, vy) 的 RMSE"""
    err = est_states[:, 2:4] - true_states[:, 2:4]
    return float(np.sqrt(np.mean(err ** 2)))


def compute_nees(est_states, true_states, P_history):
    """归一化估计误差平方（NEES），仅计算位置维度（索引 0, 1）。

    Parameters
    ----------
    est_states : array (T, 6)
    true_states : array (T, 6)
    P_history : list of (6, 6) arrays, length T
        每一步的协方差矩阵。

    Returns
    -------
    float — 平均 NEES（理想值应接近 2，即位置维度数）
    """
    nees_vals = []
    for t in range(min(len(est_states), len(P_history))):
        err = (est_states[t, :2] - true_states[t, :2]).astype(np.float64)
        P_pos = P_history[t][:2, :2].astype(np.float64)
        try:
            P_inv = np.linalg.inv(P_pos)
        except np.linalg.LinAlgError:
            P_inv = np.linalg.pinv(P_pos)
        nees_vals.append(float(err @ P_inv @ err))
    return float(np.mean(nees_vals)) if nees_vals else float('nan')


def compute_mode_switch_delay(est_modes, true_modes):
    """计算模式切换检测延迟（步数）。

    Parameters
    ----------
    est_modes : array-like (T,)  估计的模式序列（整数）
    true_modes : array-like (T,)  真实模式序列（整数）

    Returns
    -------
    float — 平均切换延迟（步数）；若无切换则返回 0。
    """
    est_modes = np.asarray(est_modes)
    true_modes = np.asarray(true_modes)
    T = min(len(est_modes), len(true_modes))

    delays = []
    i = 1
    while i < T:
        if true_modes[i] != true_modes[i - 1]:
            # 真实发生切换，寻找估计方的跟进时刻
            for j in range(i, T):
                if est_modes[j] == true_modes[i]:
                    delays.append(j - i)
                    break
            else:
                delays.append(T - i)  # 从未跟上
        i += 1

    return float(np.mean(delays)) if delays else 0.0


def print_comparison_table(metrics_dict: dict):
    """格式化打印对比表格。

    Parameters
    ----------
    metrics_dict : dict
        形如 {'Traditional IMM': {'MOTVE': 0.94, ...}, 'NeurAda-IMM': {...}, ...}
    """
    if not metrics_dict:
        return

    methods = list(metrics_dict.keys())
    all_keys = []
    for v in metrics_dict.values():
        for k in v:
            if k not in all_keys:
                all_keys.append(k)

    col_w = max(max(len(m) for m in methods), 16)
    key_w = max(max(len(k) for k in all_keys), 18)

    header = f"{'Metric':<{key_w}}" + "".join(f"  {m:>{col_w}}" for m in methods)
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for k in all_keys:
        row = f"{k:<{key_w}}"
        for m in methods:
            val = metrics_dict[m].get(k, float('nan'))
            row += f"  {val:>{col_w}.4f}"
        print(row)
    print(sep + "\n")

