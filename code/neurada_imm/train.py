import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import DEVICE, BATCH_SIZE, OBS_SIGMA, DT, CONTEXT_DIM, K_ACTIVE, L_MODELS, PROCESS_NOISE
from .datasets import TrajectoryDataset
from .models import NeuralMotionModel, ModelSelector, TransitionNet, WeightNet
from .filters import NeurAdaIMM


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
    """端到端微调选择网络、转移网络、权重网络（保持原实现）"""
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

    # 注意：这里原代码用 CrossEntropyLoss + one-hot target（并不严格正确），为保持行为一致先不改。
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
                Q=PROCESS_NOISE,
                R=np.eye(2) * OBS_SIGMA ** 2,
                K=K,
                context_dim=CONTEXT_DIM,
            )
            imm.initialize(z[0].cpu().numpy(), ctx[0].cpu().numpy())

            # 轨迹损失（原实现存在“np累积不可反传”的问题；这里保留仅作监控，不参与反传）
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
