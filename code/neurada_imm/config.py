import numpy as np
import torch

# ========================== 全局参数 ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = 0.1                     # 时间步长 (s)，用于合成数据及重采样
T_STEPS = 200                 # 合成轨迹长度
OBS_SIGMA = 0.5               # 观测噪声标准差
EPOCHS_PRETRAIN = 30          # 模型库预训练轮数
EPOCHS_FINETUNE = 20          # 端到端微调轮数
BATCH_SIZE = 64
L_MODELS = 10                  # 模型库大小
K_ACTIVE = 3                   # 每个时刻活跃模型数
CONTEXT_DIM = 5                 # 上下文维度：类别(2 one-hot) + 密度(1) + 路口(1) + 历史速度(1)

# 过程噪声协方差（位置噪声小，速度噪声大；用于修复“速度不更新/近似常数”的问题）
# 说明：只观测(x,y)时，若Q太小且P传播没有引入x-v耦合，会导致速度上的卡尔曼增益很弱。
# 调参记录（2026-03-08）：上一轮把速度过程噪声调到5.0导致vx抖动明显增大。
# 折中方案：保持速度可更新，但降低到 1.0，以减少高频抖动。
# 回滚：恢复较稳的基线噪声设定（后续再做更细的网格搜索）
PROCESS_NOISE = np.diag([0.1, 0.1, 1.0, 1.0, 0.1, 0.1]).astype(np.float32)

# 参数扫描得到的推荐设置（合成基准）：
# qv_scale=0.60, ema_lambda=0.93
QV_SCALE = 0.60
EMA_LAMBDA = 0.93

# nuScenes
NUSCENES_DATAROOT = r"C:\\Users\\ZhuJiaLe\\Desktop\\IMM学习\\data\\v1.0-mini"
NUSCENES_VERSION = "v1.0-mini"
