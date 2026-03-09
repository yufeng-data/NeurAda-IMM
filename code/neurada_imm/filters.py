import numpy as np
import torch
from filterpy.kalman import KalmanFilter

from .config import DEVICE


def _safe_log_likelihood(innov: np.ndarray, S: np.ndarray) -> float:
    """使用数值稳定的对数似然计算。

    利用 slogdet 代替直接计算行列式，避免 det 为 0 或极小时出现 inf/nan。
    同时对马氏距离设置上限，防止 exp 溢出。
    """
    _INVALID_LOG_LIKELIHOOD = -1e18
    _MAX_MAHALANOBIS_SQUARED = 200.0

    n = len(innov)
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        # S 不正定：退回极小的对数似然
        return _INVALID_LOG_LIKELIHOOD

    # 马氏距离平方，设上限防止 exp(-很大) -> 0 被误判
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    maha2 = float(innov @ S_inv @ innov)
    maha2 = min(maha2, _MAX_MAHALANOBIS_SQUARED)

    log_like = -0.5 * (n * np.log(2 * np.pi) + logdet + maha2)
    return float(log_like)


def _log_likes_to_likes(log_likes: np.ndarray) -> np.ndarray:
    """用 log-sum-exp 技巧将对数似然向量转换为似然向量（归一化到最大值为 0）。

    避免直接 exp(large_negative) -> 0 或 exp(large_positive) -> inf。
    """
    log_likes = np.asarray(log_likes, dtype=np.float64)
    max_ll = np.max(log_likes)
    likes = np.exp(log_likes - max_ll)
    return likes.astype(np.float32)


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

        # 速度稳定化补丁（默认关闭）：会改变滤波器统计一致性，可能造成性能回退
        self._enable_vel_postprocess = False
        self._vel_smooth = 0.8  # 越大越平滑
        self._vel_gain = 0.15   # 位置创新到速度的注入增益

    def initialize(self, meas):
        self.x = np.array([meas[0], meas[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.P = (np.eye(6, dtype=np.float32) * 10.0)

    def predict(self):
        # 1) 协方差预测（CV线性近似）
        F = np.eye(6, dtype=np.float32)
        F[0, 2] = self.dt
        F[1, 3] = self.dt
        self.P = F @ self.P @ F.T + self.Q

        # 2) 状态预测（NN）
        s_tensor = torch.FloatTensor(self.x).unsqueeze(0)
        with torch.no_grad():
            x_pred = self.nn(s_tensor).squeeze().cpu().numpy()
        self.x = x_pred.astype(np.float32)

    def update(self, z):
        z = np.asarray(z, dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y

        # Joseph form：保证 P 对称正定，即使 K 不精确
        I_KH = np.eye(6, dtype=np.float32) - K @ self.H
        self.P = (I_KH @ self.P @ I_KH.T + K @ self.R @ K.T).astype(np.float32)

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

        log_likes = np.zeros(K, dtype=np.float64)
        for j in range(K):
            self.models[j].x = mixed_x[j]
            self.models[j].P = mixed_P[j]
            self.models[j].predict()
            self.models[j].update(z.reshape(2, 1))

            innov = z - self.models[j].H @ self.models[j].x.flatten()
            S = self.models[j].H @ self.models[j].P @ self.models[j].H.T + self.R
            log_likes[j] = _safe_log_likelihood(innov, S)

        likelihood = _log_likes_to_likes(log_likes)

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

        # 权重平滑：减少在线抖动（尤其体现在vx/vy）
        try:
            from .config import EMA_LAMBDA
            self._ema_lambda = float(EMA_LAMBDA)
        except Exception:
            self._ema_lambda = 0.9
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

        log_likes = np.zeros(K, dtype=np.float64)
        for j in range(K):
            self.models[j].x = mixed_x[j]
            self.models[j].P = mixed_P[j]
            self.models[j].predict()
            innov, S, detS = self.models[j].update(z)
            log_likes[j] = _safe_log_likelihood(innov, S)

        likelihood = _log_likes_to_likes(log_likes)

        c = likelihood @ cbar
        self.mu = (cbar * likelihood) / (c + 1e-8)
        self.model_probs_history.append(self.mu.copy())
        self.selected_history.append(self.active_indices.copy())

        states = np.stack([m.x for m in self.models], axis=0)
        with torch.no_grad():
            states_t = torch.FloatTensor(states).unsqueeze(0).to(DEVICE)
            mu_t = torch.FloatTensor(self.mu).unsqueeze(0).to(DEVICE)
            weights = self.weight_net(states_t, mu_t).squeeze().cpu().numpy()

        # EMA 平滑权重，避免每步权重剧烈跳变
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
