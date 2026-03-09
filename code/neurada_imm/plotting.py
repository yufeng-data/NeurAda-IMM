import numpy as np
import matplotlib.pyplot as plt

from .config import DT


def plot_tracking_comparison(true_states, noisy_meas, trad_est, neur_est, out_path: str = 'tracking_comparison.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 上左：轨迹
    ax = axes[0, 0]
    ax.plot(true_states[:, 0], true_states[:, 1], 'k-', label='True')
    ax.plot(noisy_meas[:, 0], noisy_meas[:, 1], 'k.', markersize=1, label='Measurements')
    ax.plot(trad_est[:, 0], trad_est[:, 1], 'b--', label='Traditional IMM')
    ax.plot(neur_est[:, 0], neur_est[:, 1], 'r-.', label='NeurAda-IMM')
    ax.legend()
    ax.set_title('Trajectory Comparison')
    ax.axis('equal')
    ax.grid(True)

    # 上右：vx 速度曲线
    time_axis = np.arange(len(true_states)) * DT
    ax = axes[0, 1]
    ax.plot(time_axis, true_states[:, 2], 'k-', label='True vx')
    ax.plot(time_axis, trad_est[:, 2], 'b--', label='Trad vx')
    ax.plot(time_axis, neur_est[:, 2], 'r-.', label='Neur vx')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('vx (m/s)')
    ax.legend()
    ax.set_title('Velocity X')
    ax.grid(True)

    # 下左：vy 速度曲线
    ax = axes[1, 0]
    ax.plot(time_axis, true_states[:, 3], 'k-', label='True vy')
    ax.plot(time_axis, trad_est[:, 3], 'b--', label='Trad vy')
    ax.plot(time_axis, neur_est[:, 3], 'r-.', label='Neur vy')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('vy (m/s)')
    ax.legend()
    ax.set_title('Velocity Y')
    ax.grid(True)

    # 下右：位置误差时间曲线
    ax = axes[1, 1]
    pos_err_trad = np.sqrt(np.sum((trad_est[:, :2] - true_states[:, :2]) ** 2, axis=1))
    pos_err_neur = np.sqrt(np.sum((neur_est[:, :2] - true_states[:, :2]) ** 2, axis=1))
    ax.plot(time_axis, pos_err_trad, 'b--', label='Traditional IMM')
    ax.plot(time_axis, pos_err_neur, 'r-.', label='NeurAda-IMM')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (m)')
    ax.legend()
    ax.set_title('Position Error over Time')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"图像已保存至: {out_path}")
    plt.show()


def plot_model_probs(probs_history, out_path: str = 'model_probs.png'):
    """绘制模型概率随时间变化。

    Parameters
    ----------
    probs_history : list of arrays (K,), length T
    out_path : str
    """
    probs = np.array(probs_history)  # (T, K)
    T, K = probs.shape
    time_axis = np.arange(T) * DT

    plt.figure(figsize=(10, 4))
    for k in range(K):
        plt.plot(time_axis, probs[:, k], label=f'Model {k}')
    plt.xlabel('Time (s)')
    plt.ylabel('Model Probability')
    plt.title('Model Probabilities over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"图像已保存至: {out_path}")
    plt.show()


def plot_nuscenes_tracking(true_states, noisy_meas, trad_est, neur_est, out_path: str = 'nuscenes_tracking.png'):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左：轨迹
    ax = axes[0]
    ax.plot(true_states[:, 0], true_states[:, 1], 'k-', linewidth=2, label='True')
    ax.plot(noisy_meas[:, 0], noisy_meas[:, 1], 'k.', markersize=2, alpha=0.5, label='Measurements')
    ax.plot(trad_est[:, 0], trad_est[:, 1], 'b--', linewidth=1.5, label='Traditional IMM')
    ax.plot(neur_est[:, 0], neur_est[:, 1], 'r-.', linewidth=1.5, label='NeurAda-IMM')
    ax.legend()
    ax.set_title('nuScenes Trajectory Tracking')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.axis('equal')
    ax.grid(True)

    # 右：位置误差时间曲线
    ax = axes[1]
    time_axis = np.arange(len(true_states)) * DT
    pos_err_trad = np.sqrt(np.sum((trad_est[:, :2] - true_states[:, :2]) ** 2, axis=1))
    pos_err_neur = np.sqrt(np.sum((neur_est[:, :2] - true_states[:, :2]) ** 2, axis=1))
    ax.plot(time_axis, pos_err_trad, 'b--', label='Traditional IMM')
    ax.plot(time_axis, pos_err_neur, 'r-.', label='NeurAda-IMM')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Position Error over Time')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"图像已保存至: {out_path}")
    plt.show()

