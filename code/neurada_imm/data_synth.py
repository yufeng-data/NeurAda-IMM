import numpy as np

from .config import DT, T_STEPS, OBS_SIGMA, CONTEXT_DIM


def generate_trajectory_with_modes(num_steps: int = T_STEPS, dt: float = DT, noise_std: float = 0.1, density_level: str = 'medium'):
    """生成带真实运动模式标签的轨迹。

    模式:
      0-匀速直线, 1-匀速转弯, 2-匀加速, 3-静止

    返回:
      states: (num_steps, 6) [x,y,vx,vy,ax,ay]
      modes:  (num_steps,)
    """
    if density_level == 'sparse':
        base = [0] * 80 + [1] * 40 + [0] * 80
    elif density_level == 'medium':
        base = [0] * 50 + [1] * 40 + [3] * 20 + [0] * 50 + [1] * 40
    else:  # dense
        base = [0] * 30 + [2] * 20 + [1] * 30 + [3] * 10 + [0] * 30 + [1] * 30 + [2] * 20 + [0] * 30

    modes = (base * (num_steps // len(base) + 1))[:num_steps]

    state = np.array([0., 0., 5., 0., 0., 0.], dtype=np.float32)
    states = [state.copy()]
    true_modes = [modes[0]]

    for t in range(1, num_steps):
        mode = modes[t]
        x, y, vx, vy, ax, ay = state

        if mode == 0:  # 匀速直线
            new_x = x + vx * dt
            new_y = y + vy * dt
            new_vx, new_vy = vx, vy
        elif mode == 1:  # 匀速转弯
            omega = 1.0
            theta = np.arctan2(vy, vx) + omega * dt
            speed = np.linalg.norm([vx, vy])
            new_vx = speed * np.cos(theta)
            new_vy = speed * np.sin(theta)
            new_x = x + vx * dt
            new_y = y + vy * dt
        elif mode == 2:  # 匀加速
            acc = 2.0
            direction = np.array([vx, vy]) / (np.linalg.norm([vx, vy]) + 1e-6)
            new_vx = vx + acc * direction[0] * dt
            new_vy = vy + acc * direction[1] * dt
            new_x = x + vx * dt + 0.5 * acc * direction[0] * dt ** 2
            new_y = y + vy * dt + 0.5 * acc * direction[1] * dt ** 2
        else:  # 静止
            new_x, new_y = x, y
            new_vx, new_vy = 0., 0.

        new_state = np.array([new_x, new_y, new_vx, new_vy, 0., 0.], dtype=np.float32)
        new_state += np.random.normal(0, noise_std, size=6).astype(np.float32)

        states.append(new_state)
        true_modes.append(mode)
        state = new_state

    return np.asarray(states, dtype=np.float32), np.asarray(true_modes, dtype=np.int64)


def generate_context(modes, density_value: float = 0.5):
    """根据真实模式生成上下文特征。返回 (T, CONTEXT_DIM)。"""
    num_steps = len(modes)
    category = np.random.choice([0, 1])
    cat_onehot = np.zeros(2, dtype=np.float32)
    cat_onehot[category] = 1.0

    density = np.full(num_steps, density_value, dtype=np.float32) + np.random.normal(0, 0.05, num_steps).astype(np.float32)
    density = np.clip(density, 0, 1)

    intersection = np.zeros(num_steps, dtype=np.float32)
    for i in range(1, num_steps):
        if modes[i] != modes[i - 1] and np.random.rand() > 0.3:
            intersection[i] = 1
            if i + 1 < num_steps:
                intersection[i + 1] = 1

    hist_speed = np.zeros(num_steps, dtype=np.float32)  # 将在滤波过程中填充

    context = np.zeros((num_steps, CONTEXT_DIM), dtype=np.float32)
    context[:, :2] = cat_onehot
    context[:, 2] = density
    context[:, 3] = intersection
    context[:, 4] = hist_speed
    return context


def add_measurement_noise(states, sigma: float = OBS_SIGMA):
    noisy = states.copy()
    noisy[:, :2] += np.random.normal(0, sigma, size=states[:, :2].shape).astype(np.float32)
    return noisy
