import numpy as np
from scipy import interpolate

try:
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False


def load_nuscenes_trajectory(nusc, scene_idx: int = 0, category_name: str = 'human.pedestrian.adult', dt: float = 0.1):
    """从 nuScenes 提取指定场景的第一个目标物体(x,y)轨迹，并重采样到固定 dt。

    返回:
      states: (T,6) [x,y,vx,vy,ax,ay]
      t_new:  (T,)
    """
    if not NUSCENES_AVAILABLE:
        raise RuntimeError("未安装 nuscenes-devkit / pyquaternion，无法加载 nuScenes 轨迹")

    scene = nusc.scene[scene_idx]
    sample_token = scene['first_sample_token']
    samples = []
    while sample_token != '':
        sample = nusc.get('sample', sample_token)
        samples.append(sample)
        sample_token = sample['next']

    positions = []
    timestamps = []
    for sample in samples:
        found = False
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            if ann['category_name'] == category_name:
                box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
                positions.append(box.center[:2])
                timestamps.append(sample['timestamp'] * 1e-6)
                found = True
                break
        if not found:
            if len(positions) > 0:
                positions.append(positions[-1])
                timestamps.append(timestamps[-1] + 0.5)
            else:
                continue

    if len(positions) == 0:
        raise ValueError(f"场景 {scene['name']} 中未找到类别 {category_name}")

    positions = np.asarray(positions, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float64)

    t_start, t_end = timestamps[0], timestamps[-1]
    t_new = np.arange(t_start, t_end, dt)
    if len(t_new) < 2:
        t_new = np.array([t_start, t_start + dt])

    pos_interp = interpolate.interp1d(timestamps, positions, axis=0, kind='linear', fill_value='extrapolate')
    pos_new = pos_interp(t_new).astype(np.float32)

    vx = np.gradient(pos_new[:, 0], dt).astype(np.float32)
    vy = np.gradient(pos_new[:, 1], dt).astype(np.float32)
    ax = np.gradient(vx, dt).astype(np.float32)
    ay = np.gradient(vy, dt).astype(np.float32)

    states = np.zeros((len(t_new), 6), dtype=np.float32)
    states[:, 0] = pos_new[:, 0]
    states[:, 1] = pos_new[:, 1]
    states[:, 2] = vx
    states[:, 3] = vy
    states[:, 4] = ax
    states[:, 5] = ay

    return states, t_new
