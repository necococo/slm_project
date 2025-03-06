import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    カスタム実装のSwiGLU活性化関数
    形式: SwiGLU(x) = x * SiLU(gate(x))
    
    参照: GLU Variants Improve Transformer (https://arxiv.org/abs/2002.05202)
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        x, gate = x.chunk(2, dim=self.dim)
        return x * F.silu(gate)
