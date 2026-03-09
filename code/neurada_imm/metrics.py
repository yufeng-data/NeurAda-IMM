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
