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

class GatedMLP(nn.Module):
    """
    SwiGLUをベースにした高効率ゲート付きMLP
    これにより通常のMLPより小さいパラメータで良い性能を発揮
    """
    def __init__(self, in_features, hidden_ratio=2.6667):
        """
        Args:
            in_features: 入力次元
            hidden_ratio: 中間層の拡大率 
        """
        super().__init__()
        hidden_features = int(in_features * hidden_ratio * 2)  # *2はSwiGLUのため
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = SwiGLU()
        self.fc2 = nn.Linear(hidden_features // 2, in_features)  # SwiGLUで半分になるので調整
        
        self.init_weights()
        
    def init_weights(self):
        # 効率的な初期化
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
