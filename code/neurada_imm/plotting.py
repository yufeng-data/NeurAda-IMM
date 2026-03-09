import numpy as np
import matplotlib.pyplot as plt

from .config import DT


def plot_tracking_comparison(true_states, noisy_meas, trad_est, neur_est, out_path: str = 'tracking_comparison.png'):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(true_states[:, 0], true_states[:, 1], 'k-', label='True')
    plt.plot(noisy_meas[:, 0], noisy_meas[:, 1], 'k.', markersize=1, label='Measurements')
    plt.plot(trad_est[:, 0], trad_est[:, 1], 'b--', label='Traditional IMM')
    plt.plot(neur_est[:, 0], neur_est[:, 1], 'r-.', label='NeurAda-IMM')
    plt.legend()
    plt.title('Trajectory Comparison')
    plt.axis('equal')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    time_axis = np.arange(len(true_states)) * DT
    plt.plot(time_axis, true_states[:, 2], 'k-', label='True vx')
    plt.plot(time_axis, trad_est[:, 2], 'b--', label='Trad vx')
    plt.plot(time_axis, neur_est[:, 2], 'r-.', label='Neur vx')
    plt.xlabel('Time (s)')
    plt.ylabel('vx (m/s)')
    plt.legend()
    plt.title('Velocity X')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()


def plot_nuscenes_tracking(true_states, noisy_meas, trad_est, neur_est, out_path: str = 'nuscenes_tracking.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(true_states[:, 0], true_states[:, 1], 'k-', linewidth=2, label='True')
    plt.plot(noisy_meas[:, 0], noisy_meas[:, 1], 'k.', markersize=2, alpha=0.5, label='Measurements')
    plt.plot(trad_est[:, 0], trad_est[:, 1], 'b--', linewidth=1.5, label='Traditional IMM')
    plt.plot(neur_est[:, 0], neur_est[:, 1], 'r-.', linewidth=1.5, label='NeurAda-IMM')
    plt.legend()
    plt.title('nuScenes Trajectory Tracking')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
