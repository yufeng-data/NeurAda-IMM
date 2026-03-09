"""
NeurAda-IMM 完整实现（单文件整合版）
包含：配置、数据生成、数据集、神经网络模型、滤波器、评估指标、训练函数、绘图及主流程。
可直接运行进行合成数据实验，若已安装 nuScenes-devkit 则会自动进行真实数据测试。
"""

# ==================== 标准库与第三方库导入 ====================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy import interpolate
from filterpy.kalman import KalmanFilter

# 可选 nuScenes 依赖
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False
    print("警告：未安装 nuscenes-devkit / pyquaternion，无法加载真实数据。")

# ==================== 全局参数（原 config.py）====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = 0.1                     # 时间步长 (s)
T_STEPS = 200                 # 合成轨迹长度
OBS_SIGMA = 0.5               # 观测噪声标准差
EPOCHS_PRETRAIN = 30          # 模型库预训练轮数
EPOCHS_FINETUNE = 20          # 端到端微调轮数
BATCH_SIZE = 64
L_MODELS = 10                  # 模型库大小
K_ACTIVE = 3                   # 每个时刻活跃模型数
CONTEXT_DIM = 5                 # 上下文维度：类别(2 one-hot) + 密度(1) + 路口(1) + 历史速度(1)

# nuScenes 配置（如果安装了 devkit 且希望运行真实数据）
NUSCENES_DATAROOT = r"C:\Users\ZhuJiaLe\Desktop\IMM学习\data\v1.0-mini"
NUSCENES_VERSION = "v1.0-mini"

# 过程噪声协方差
PROCESS_NOISE = np.diag([0.1, 0.1, 1.0, 1.0, 0.1, 0.1]).astype(np.float32)

# 参数扫得到的推荐值：缩放 Q 中 vx/vy 维度的过程噪声
QV_SCALE = 0.60


def _make_Q(base_Q: np.ndarray, qv_scale: float = QV_SCALE) -> np.ndarray:
    """基于 base_Q 生成本次实验使用的 Q，并对 vx/vy 方向做缩放。"""
    Q = np.array(base_Q, dtype=np.float32, copy=True)
    if Q.shape[0] >= 4 and Q.shape[1] >= 4:
        Q[2, 2] = float(Q[2, 2]) * float(qv_scale)
        Q[3, 3] = float(Q[3, 3]) * float(qv_scale)
    return Q


# ==================== 数据生成（原 data_synth.py）====================
def generate_trajectory_with_modes(num_steps: int = T_STEPS, dt: float = DT,
                                   noise_std: float = 0.1, density_level: str = 'medium'):
    """生成带真实运动模式标签的轨迹。模式: 0-匀速直线, 1-匀速转弯, 2-匀加速, 3-静止"""
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
    """根据真实模式生成上下文特征。返回 (T, CONTEXT_DIM)"""
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


# ==================== nuScenes 数据加载（原 data_nuscenes.py）====================
def load_nuscenes_trajectory(nusc, scene_idx: int = 0, category_name: str = 'human.pedestrian.adult', dt: float = 0.1):
    """从 nuScenes 提取指定场景的第一个目标物体(x,y)轨迹，并重采样到固定 dt。"""
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


# ==================== PyTorch 数据集（原 datasets.py）====================
class TrajectoryDataset(Dataset):
    """预训练数据集：输入当前状态，输出下一状态"""
    def __init__(self, trajectories):
        self.X = []
        self.Y = []
        for traj in trajectories:
            for t in range(len(traj) - 1):
                self.X.append(traj[t])
                self.Y.append(traj[t + 1])
        self.X = np.asarray(self.X, dtype=np.float32)
        self.Y = np.asarray(self.Y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.Y[idx])


class FinetuneDataset(Dataset):
    """微调数据集：返回整条轨迹、上下文、模式列表"""
    def __init__(self, trajectories, contexts, modes_list):
        self.trajs = [torch.FloatTensor(t) for t in trajectories]
        self.ctxs = [torch.FloatTensor(c) for c in contexts]
        self.modes = [torch.as_tensor(m, dtype=torch.long) for m in modes_list]

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx], self.ctxs[idx], self.modes[idx]


# ==================== 神经网络模型（原 models.py）====================
class NeuralMotionModel(nn.Module):
    """单个运动模型（物理先验+残差学习）"""
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 输出位置和速度的增量
        )
        self.register_buffer('dt', torch.tensor(DT))

    def forward(self, s):
        vel = s[:, 2:4]
        delta_phy = torch.zeros_like(s[:, :4])
        delta_phy[:, 0] = vel[:, 0] * self.dt
        delta_phy[:, 1] = vel[:, 1] * self.dt
        delta_res = self.net(vel)
        delta = delta_phy + delta_res
        next_s = torch.cat([s[:, :4] + delta, s[:, 4:]], dim=1)
        return next_s


class ModelSelector(nn.Module):
    """选择网络：上下文 -> 对L个模型的软选择概率"""
    def __init__(self, context_dim: int, L: int, hidden_dims=None):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        layers = []
        in_dim = context_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, L))
        self.net = nn.Sequential(*layers)

    def forward(self, context):
        logits = self.net(context)
        return torch.softmax(logits, dim=-1)


class TransitionNet(nn.Module):
    """转移网络：输入当前概率和上下文，输出KxK转移矩阵"""
    def __init__(self, K: int, context_dim: int, hidden_dims=None):
        super().__init__()
        self.K = K
        hidden_dims = hidden_dims or [64, 32]
        in_dim = K + context_dim
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, K * K))
        self.net = nn.Sequential(*layers)

    def forward(self, mu, context):
        x = torch.cat([mu, context], dim=-1)
        logits = self.net(x).view(-1, self.K, self.K)
        return torch.softmax(logits, dim=-1)


class WeightNet(nn.Module):
    """自适应权重网络：输入各模型状态和概率，输出调整后的权重"""
    def __init__(self, K: int, state_dim: int = 6, hidden_dim: int = 32):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(K * (state_dim + 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K),
        )

    def forward(self, states, mu):
        batch = states.shape[0]
        feat = torch.cat([states.view(batch, -1), mu], dim=-1)
        logits = self.net(feat)
        return torch.softmax(logits, dim=-1)


# ==================== 滤波器类（原 filters.py）====================
class NNKFWrapper:
    """将神经网络模型包装成类似卡尔曼滤波器的接口（修复协方差传播）"""
    def __init__(self, nn_model, dt: float, Q: np.ndarray, R: np.ndarray):
        self.nn = nn_model
        self.dt = float(dt)
        self.Q = Q.astype(np.float32)
        self.R = R.astype(np.float32)
        self.H = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0]],
            dtype=np.float32,
        )
        self.x = None
        self.P = None
        self._enable_vel_postprocess = False
        self._vel_smooth = 0.8
        self._vel_gain = 0.15

    def initialize(self, meas):
        self.x = np.array([meas[0], meas[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 10.0

    def predict(self):
        F = np.eye(6, dtype=np.float32)
        F[0, 2] = self.dt
        F[1, 3] = self.dt
        self.P = F @ self.P @ F.T + self.Q

        s_tensor = torch.FloatTensor(self.x).unsqueeze(0)
        with torch.no_grad():
            x_pred = self.nn(s_tensor).squeeze().cpu().numpy()
        self.x = x_pred.astype(np.float32)

    def update(self, z):
        z = np.asarray(z, dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6, dtype=np.float32) - K @ self.H) @ self.P

        if self._enable_vel_postprocess:
            dv = (self._vel_gain * y / max(self.dt, 1e-6)).astype(np.float32)
            v_new = self.x[2:4] + dv
            self.x[2:4] = (self._vel_smooth * self.x[2:4] + (1.0 - self._vel_smooth) * v_new).astype(np.float32)

        detS = float(np.linalg.det(S))
        return y, S, detS


class TraditionalIMM:
    """传统IMM，固定两个模型：CV和CTRV（这里转弯模型仅用不同Q近似）"""
    def __init__(self, dt: float, Q_cv: np.ndarray, Q_ctrv: np.ndarray, R: np.ndarray, trans_matrix: np.ndarray):
        self.dt = float(dt)
        self.Q_cv = Q_cv
        self.Q_ctrv = Q_ctrv
        self.R = R
        self.Pi = trans_matrix
        self.mu = np.array([0.7, 0.3], dtype=np.float32)
        self.models = self._create_models()
        self.x_combined = None

    def _create_models(self):
        models = []
        for Q in [self.Q_cv, self.Q_ctrv]:
            kf = KalmanFilter(dim_x=6, dim_z=2)
            kf.F = np.eye(6)
            kf.F[0, 2] = self.dt
            kf.F[1, 3] = self.dt
            kf.H = np.zeros((2, 6))
            kf.H[0, 0] = 1
            kf.H[1, 1] = 1
            kf.Q = Q
            kf.R = self.R
            kf.P = np.eye(6) * 10
            models.append(kf)
        return models

    def initialize(self, meas):
        for m in self.models:
            m.x = np.array([[meas[0]], [meas[1]], [0], [0], [0], [0]])
        self.x_combined = self.models[0].x.flatten()

    def step(self, z):
        K = 2
        cbar = self.mu @ self.Pi
        mu_cond = np.zeros((K, K), dtype=np.float32)
        for j in range(K):
            mu_cond[:, j] = self.mu * self.Pi[:, j] / (cbar[j] + 1e-8)

        mixed_x = []
        mixed_P = []
        for j in range(K):
            xj = np.zeros((6, 1))
            Pj = np.zeros((6, 6))
            for i in range(K):
                xj += mu_cond[i, j] * self.models[i].x
            for i in range(K):
                diff = self.models[i].x - xj
                Pj += mu_cond[i, j] * self.models[i].P + diff @ diff.T
            mixed_x.append(xj)
            mixed_P.append(Pj)

        likelihood = np.zeros(K, dtype=np.float32)
        for j in range(K):
            self.models[j].x = mixed_x[j]
            self.models[j].P = mixed_P[j]
            self.models[j].predict()
            self.models[j].update(z.reshape(2, 1))

            innov = z - self.models[j].H @ self.models[j].x.flatten()
            S = self.models[j].H @ self.models[j].P @ self.models[j].H.T + self.R
            detS = np.linalg.det(S)
            like = (1.0 / np.sqrt((2 * np.pi) ** 2 * detS)) * np.exp(-0.5 * innov @ np.linalg.inv(S) @ innov)
            likelihood[j] = like

        c = likelihood @ cbar
        self.mu = (cbar * likelihood) / (c + 1e-8)

        x_comb = np.zeros(6)
        for j in range(K):
            x_comb += self.mu[j] * self.models[j].x.flatten()
        self.x_combined = x_comb
        return x_comb


class NeurAdaIMM:
    """NeurAda-IMM：集成神经网络组件"""
    def __init__(self, dt, model_bank, selector, trans_net, weight_net, Q, R, K=3, context_dim=5):
        self.dt = float(dt)
        self.model_bank = model_bank
        self.L = len(model_bank)
        self.K = min(K, self.L)
        self.selector = selector
        self.trans_net = trans_net
        self.weight_net = weight_net
        self.Q = Q
        self.R = R
        self.context_dim = context_dim

        self.active_indices = None
        self.models = []
        self.mu = np.ones(self.K, dtype=np.float32) / self.K
        self.x_combined = None
        self.model_probs_history = []
        self.selected_history = []
        # 参数扫最优：EMA_LAMBDA = 0.93
        self._ema_lambda = 0.93
        self._w_ema = None

    def initialize(self, meas, context):
        with torch.no_grad():
            ctx_t = torch.FloatTensor(context).unsqueeze(0).to(DEVICE)
            probs = self.selector(ctx_t).squeeze().cpu().numpy()

        self.active_indices = np.argsort(probs)[-self.K:]
        self.models = []
        for idx in self.active_indices:
            nn_model = self.model_bank[int(idx)]
            wrapper = NNKFWrapper(nn_model, self.dt, self.Q, self.R)
            wrapper.initialize(meas)
            self.models.append(wrapper)

        self.mu = np.ones(self.K, dtype=np.float32) / self.K
        self.x_combined = self.models[0].x.copy()
        self._w_ema = None

    def step(self, z, context):
        K = self.K
        with torch.no_grad():
            ctx_t = torch.FloatTensor(context).unsqueeze(0).to(DEVICE)
            mu_t = torch.FloatTensor(self.mu).unsqueeze(0).to(DEVICE)
            trans = self.trans_net(mu_t, ctx_t).squeeze().cpu().numpy()  # (K,K)

        cbar = self.mu @ trans
        mu_cond = np.zeros((K, K), dtype=np.float32)
        for j in range(K):
            mu_cond[:, j] = self.mu * trans[:, j] / (cbar[j] + 1e-8)

        mixed_x = []
        mixed_P = []
        for j in range(K):
            xj = np.zeros(6, dtype=np.float32)
            Pj = np.zeros((6, 6), dtype=np.float32)
            for i in range(K):
                xj += mu_cond[i, j] * self.models[i].x
            for i in range(K):
                diff = self.models[i].x - xj
                Pj += mu_cond[i, j] * (self.models[i].P + np.outer(diff, diff))
            mixed_x.append(xj)
            mixed_P.append(Pj)

        likelihood = np.zeros(K, dtype=np.float32)
        for j in range(K):
            self.models[j].x = mixed_x[j]
            self.models[j].P = mixed_P[j]
            self.models[j].predict()
            innov, S, detS = self.models[j].update(z)
            like = (1.0 / np.sqrt((2 * np.pi) ** 2 * detS)) * np.exp(-0.5 * innov @ np.linalg.inv(S) @ innov)
            likelihood[j] = like

        c = likelihood @ cbar
        self.mu = (cbar * likelihood) / (c + 1e-8)
        self.model_probs_history.append(self.mu.copy())
        self.selected_history.append(self.active_indices.copy())

        states = np.stack([m.x for m in self.models], axis=0)
        with torch.no_grad():
            states_t = torch.FloatTensor(states).unsqueeze(0).to(DEVICE)
            mu_t = torch.FloatTensor(self.mu).unsqueeze(0).to(DEVICE)
            weights = self.weight_net(states_t, mu_t).squeeze().cpu().numpy()

        if self._w_ema is None:
            self._w_ema = weights.astype(np.float32)
        else:
            self._w_ema = (self._ema_lambda * self._w_ema + (1.0 - self._ema_lambda) * weights).astype(np.float32)
        weights = self._w_ema

        x_comb = np.zeros(6, dtype=np.float32)
        for j in range(K):
            x_comb += weights[j] * self.models[j].x
        self.x_combined = x_comb
        return x_comb


# ==================== 评估指标（原 metrics.py）====================
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


# ==================== 训练函数（原 train.py）====================
def pretrain_model_bank(trajectories, L: int = L_MODELS, epochs: int = 30):
    """预训练模型库，使用自监督 + 多样性正则"""
    dataset = TrajectoryDataset(trajectories)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model_bank = [NeuralMotionModel().to(DEVICE) for _ in range(L)]
    optimizers = [optim.Adam(m.parameters(), lr=0.001) for m in model_bank]
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            for opt in optimizers:
                opt.zero_grad()

            preds = [model(batch_x) for model in model_bank]

            for i, model in enumerate(model_bank):
                pred = preds[i]
                mse = criterion(pred[:, :4], batch_y[:, :4])

                if L > 1:
                    other_preds = torch.stack([preds[j].detach() for j in range(L) if j != i], dim=0)
                    pred_expanded = pred[:, :2].unsqueeze(0).expand(L - 1, -1, -1)
                    cos = nn.functional.cosine_similarity(pred_expanded, other_preds[:, :, :2], dim=-1).mean()
                    div_loss = cos
                else:
                    div_loss = 0.0

                loss = mse + 0.1 * div_loss
                loss.backward()
                total_loss += float(loss.item())

            for opt in optimizers:
                opt.step()

        print(f"Pretrain Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

    return model_bank


def finetune_adaptive_nets(model_bank, dataloader, epochs: int = 20):
    """端到端微调选择网络、转移网络、权重网络"""
    L = len(model_bank)
    K = K_ACTIVE
    context_dim = CONTEXT_DIM

    selector = ModelSelector(context_dim, L).to(DEVICE)
    trans_net = TransitionNet(K, context_dim).to(DEVICE)
    weight_net = WeightNet(K).to(DEVICE)

    optimizer = optim.Adam(
        list(selector.parameters()) + list(trans_net.parameters()) + list(weight_net.parameters()),
        lr=0.001,
    )

    ce_loss = nn.CrossEntropyLoss()

    for m in model_bank:
        for p in m.parameters():
            p.requires_grad = False

    mode_groups = {
        0: list(range(L // 2)),
        1: list(range(L // 2, L)),
        2: list(range(L // 2, L)),
        3: [0],
    }

    for epoch in range(epochs):
        total_loss = 0.0

        for traj_batch, ctx_batch, modes_batch in dataloader:
            traj = traj_batch[0].to(DEVICE)
            ctx = ctx_batch[0].to(DEVICE)
            modes = modes_batch[0].to(DEVICE)

            T = traj.shape[0]
            z = traj[:, :2] + torch.randn_like(traj[:, :2]) * OBS_SIGMA

            imm = NeurAdaIMM(
                DT,
                model_bank,
                selector,
                trans_net,
                weight_net,
                Q=_make_Q(PROCESS_NOISE, QV_SCALE),
                R=np.eye(2) * OBS_SIGMA ** 2,
                K=K,
                context_dim=CONTEXT_DIM,
            )
            imm.initialize(z[0].cpu().numpy(), ctx[0].cpu().numpy())

            # 轨迹损失仅作监控，不参与反传
            _loss_seq = 0.0
            for t in range(1, T):
                x_pred = imm.step(z[t].cpu().numpy(), ctx[t].cpu().numpy())
                state_true = traj[t].cpu().numpy()
                _loss_seq += float(np.mean((x_pred[:4] - state_true[:4]) ** 2))

            sel_loss_total = 0.0
            for t in range(T):
                ctx_t = ctx[t].unsqueeze(0)
                sel_probs = selector(ctx_t)

                target = torch.zeros(1, L, device=DEVICE)
                mode = int(modes[t].item())
                good_indices = [i for i in mode_groups.get(mode, []) if i < L]
                if good_indices:
                    target[0, good_indices] = 1.0 / len(good_indices)
                else:
                    target[0, :L] = 1.0 / L

                sel_loss_total = sel_loss_total + ce_loss(sel_probs, target)

            optimizer.zero_grad()
            sel_loss_total.backward()
            optimizer.step()

            total_loss += float(sel_loss_total.item()) / max(T, 1)

        print(f"Finetune Epoch {epoch + 1}/{epochs}, Selection Loss: {total_loss / len(dataloader):.4f}")

    return selector, trans_net, weight_net


# ==================== 绘图函数（原 plotting.py）====================
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


# ==================== 主流程（原入口脚本）====================
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
    print(f"传统IMM     - MOTVE: {motve_trad:.4f}, 位置RMSE: {pos_rmse_trad:.4f}")
    print(f"NeurAda-IMM - MOTVE: {motve_neur:.4f}, 位置RMSE: {pos_rmse_neur:.4f}")

    plot_nuscenes_tracking(true_states, noisy_meas, trad_est, neur_est)


def main():
    print("=" * 60)
    print("NeurAda-IMM 完整仿真实验 (单文件整合版)")
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
    finetune_dataset = FinetuneDataset(train_trajs, train_ctxs, train_modes_list)
    finetune_loader = DataLoader(finetune_dataset, batch_size=1, shuffle=True)
    selector, trans_net, weight_net = finetune_adaptive_nets(model_bank, finetune_loader, epochs=EPOCHS_FINETUNE)

    print("\n生成合成测试轨迹...")
    test_traj, test_modes = generate_trajectory_with_modes(num_steps=T_STEPS, density_level='dense')
    test_ctx = generate_context(test_modes, density_value=0.8)
    noisy_meas = add_measurement_noise(test_traj)

    print("\n运行传统IMM...")
    Q_cv = np.diag([0.1, 0.1, 0.5, 0.5, 0.01, 0.01]).astype(np.float32)
    Q_ctrv = np.diag([0.1, 0.1, 1.0, 1.0, 0.1, 0.1]).astype(np.float32)
    R = (np.eye(2) * OBS_SIGMA ** 2).astype(np.float32)
    trans_matrix = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=np.float32)

    imm_trad = TraditionalIMM(DT, Q_cv, Q_ctrv, R, trans_matrix)
    imm_trad.initialize(noisy_meas[0, :2])
    trad_est = [imm_trad.x_combined.copy()]
    for t in range(1, T_STEPS):
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
    neur_imm.initialize(noisy_meas[0, :2], test_ctx[0])
    neur_est = [neur_imm.x_combined.copy()]
    for t in range(1, T_STEPS):
        neur_est.append(neur_imm.step(noisy_meas[t, :2], test_ctx[t]).copy())
    neur_est = np.asarray(neur_est, dtype=np.float32)

    print(f"NeurAda-IMM 速度列均值: vx={neur_est[:, 2].mean():.4f}, vy={neur_est[:, 3].mean():.4f}")
    print(f"传统IMM     速度列均值: vx={trad_est[:, 2].mean():.4f}, vy={trad_est[:, 3].mean():.4f}")

    motve_trad = compute_motve(trad_est, test_traj)
    motve_neur = compute_motve(neur_est, test_traj)
    pos_rmse_trad = compute_position_rmse(trad_est, test_traj)
    pos_rmse_neur = compute_position_rmse(neur_est, test_traj)
    mas = compute_mas(neur_imm.selected_history, test_modes, L_MODELS)

    print("\n合成数据测试结果：")
    print(f"传统IMM     - MOTVE: {motve_trad:.4f}, 位置RMSE: {pos_rmse_trad:.4f}")
    print(f"NeurAda-IMM - MOTVE: {motve_neur:.4f}, 位置RMSE: {pos_rmse_neur:.4f}")
    print(f"MAS: {mas:.4f}")

    plot_tracking_comparison(test_traj, noisy_meas, trad_est, neur_est)

    test_on_nuscenes(model_bank, selector, trans_net, weight_net)

    print("\n实验完成！")


if __name__ == '__main__':
    main()