import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """端到端微调选择网络、转移网络、权重网络"""
    L = len(model_bank)
    K = K_ACTIVE
    context_dim = CONTEXT_DIM

    selector = ModelSelector(context_dim, L).to(DEVICE)
    trans_net = TransitionNet(K, context_dim).to(DEVICE)
    weight_net = WeightNet(K).to(DEVICE)

    # Optimizer 1：selector 和 trans_net
    optimizer = optim.Adam(
        list(selector.parameters()) + list(trans_net.parameters()),
        lr=0.001,
    )

    # Optimizer 2：weight_net 独立可微训练路径
    weight_optimizer = optim.Adam(weight_net.parameters(), lr=0.0005)

    for m in model_bank:
        for p in m.parameters():
            p.requires_grad = False

    mode_groups = {
        0: list(range(L // 2)),
        1: list(range(L // 2, L)),
        2: list(range(L // 2, L)),
        3: [0],
    }

    mse_loss = nn.MSELoss()

    for epoch in range(epochs):
        total_sel_loss = 0.0
        total_wt_loss = 0.0

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

            for t in range(1, T):
                imm.step(z[t].cpu().numpy(), ctx[t].cpu().numpy())

            # ---- Selector loss（使用 KL 散度代替 CrossEntropyLoss）----
            sel_loss_total = torch.tensor(0.0, device=DEVICE)
            for t in range(T):
                ctx_t = ctx[t].unsqueeze(0)
                # Use raw logits + log_softmax for numerical stability
                logits = selector.net(ctx_t)  # (1, L)
                log_probs = F.log_softmax(logits, dim=-1)

                target = torch.zeros(1, L, device=DEVICE)
                mode = int(modes[t].item())
                good_indices = [i for i in mode_groups.get(mode, []) if i < L]
                if good_indices:
                    target[0, good_indices] = 1.0 / len(good_indices)
                else:
                    target[0, :L] = 1.0 / L

                sel_loss_total = sel_loss_total + F.kl_div(log_probs, target, reduction='batchmean')

            # 归一化到时间步数
            sel_loss_total = sel_loss_total / max(T, 1)

            optimizer.zero_grad()
            sel_loss_total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(selector.parameters()) + list(trans_net.parameters()),
                max_norm=5.0,
            )
            optimizer.step()
            total_sel_loss += float(sel_loss_total.item())

            # ---- WeightNet 独立可微训练路径 ----
            # 随机采样时间步，用模型库前 K 个模型预测，weight_net 加权融合后和真实下一步做 MSE
            sample_steps = min(T - 1, 16)
            step_indices = np.random.choice(T - 1, size=sample_steps, replace=False)

            wt_loss = torch.tensor(0.0, device=DEVICE)
            for t in step_indices:
                state_t = traj[t].unsqueeze(0)  # (1, 6)
                state_next_true = traj[t + 1]   # (6,)

                with torch.no_grad():
                    model_preds = torch.stack(
                        [model_bank[k](state_t) for k in range(K)], dim=1
                    )  # (1, K, 6)

                states_t = model_preds  # (1, K, 6)
                mu_uniform = torch.ones(1, K, device=DEVICE) / K
                weights = weight_net(states_t, mu_uniform)  # (1, K)

                fused = (weights.unsqueeze(-1) * model_preds).sum(dim=1).squeeze(0)  # (6,)
                wt_loss = wt_loss + mse_loss(fused[:4], state_next_true[:4])

            wt_loss = wt_loss / max(sample_steps, 1)
            weight_optimizer.zero_grad()
            wt_loss.backward()
            torch.nn.utils.clip_grad_norm_(weight_net.parameters(), max_norm=5.0)
            weight_optimizer.step()
            total_wt_loss += float(wt_loss.item())

        print(
            f"Finetune Epoch {epoch + 1}/{epochs}, "
            f"Selection Loss: {total_sel_loss / len(dataloader):.4f}, "
            f"WeightNet Loss: {total_wt_loss / len(dataloader):.4f}"
        )

    return selector, trans_net, weight_net

