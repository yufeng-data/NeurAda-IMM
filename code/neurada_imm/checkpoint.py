"""模型检查点保存与加载"""
import os
import torch

from .config import DEVICE, L_MODELS, K_ACTIVE, CONTEXT_DIM
from .models import NeuralMotionModel, ModelSelector, TransitionNet, WeightNet


def save_checkpoint(model_bank, selector, trans_net, weight_net,
                    path: str = 'checkpoint.pth', extra: dict = None):
    """保存所有模型到单个文件"""
    state = {
        'model_bank': [m.state_dict() for m in model_bank],
        'selector': selector.state_dict(),
        'trans_net': trans_net.state_dict(),
        'weight_net': weight_net.state_dict(),
        'config': {
            'L': len(model_bank),
            'K': K_ACTIVE,
            'context_dim': CONTEXT_DIM,
        },
    }
    if extra:
        state['extra'] = extra

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save(state, path)
    print(f"检查点已保存至: {path}")


def load_checkpoint(path: str = 'checkpoint.pth', device=None):
    """从文件加载所有模型"""
    device = device or DEVICE
    state = torch.load(path, map_location=device)
    cfg = state['config']
    L = cfg['L']
    K = cfg['K']
    context_dim = cfg['context_dim']

    model_bank = []
    for sd in state['model_bank']:
        m = NeuralMotionModel().to(device)
        m.load_state_dict(sd)
        m.eval()
        model_bank.append(m)

    selector = ModelSelector(context_dim, L).to(device)
    selector.load_state_dict(state['selector'])
    selector.eval()

    trans_net = TransitionNet(K, context_dim).to(device)
    trans_net.load_state_dict(state['trans_net'])
    trans_net.eval()

    weight_net = WeightNet(K).to(device)
    weight_net.load_state_dict(state['weight_net'])
    weight_net.eval()

    extra = state.get('extra', {})
    print(f"检查点已加载: {path} (L={L}, K={K})")
    return model_bank, selector, trans_net, weight_net, extra