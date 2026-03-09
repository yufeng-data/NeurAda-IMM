import numpy as np
import torch
from torch.utils.data import Dataset


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
