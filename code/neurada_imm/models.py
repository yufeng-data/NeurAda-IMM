import torch
import torch.nn as nn

from .config import DT


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
