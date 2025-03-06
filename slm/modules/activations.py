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
        # 次元が奇数の場合でも適切に処理
        dim_size = x.size(self.dim)
        if dim_size % 2 != 0:
            # 奇数次元の場合は、分割サイズを明示的に指定
            split_sizes = [dim_size // 2 + dim_size % 2, dim_size // 2]
            x, gate = torch.split(x, split_sizes, dim=self.dim)
        else:
            # 偶数次元の場合は通常通り
            x, gate = x.chunk(2, dim=self.dim)
        
        return x * F.silu(gate)

class GatedMLP(nn.Module):
    """
    SwiGLUをベースにした高効率ゲート付きMLP
    これにより通常のMLPより小さいパラメータで良い性能を発揮
    """
    def __init__(self, in_features, hidden_ratio=2):
        """
        Args:
            in_features: 入力次元
            hidden_ratio: 中間層の拡大率 
        """
        super().__init__()
        # 奇数次元の問題を避けるため、偶数に調整
        hidden_features = int(in_features * hidden_ratio * 2)
        # 偶数に強制
        if hidden_features % 2 != 0:
            hidden_features += 1
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = SwiGLU()
        # SwiGLUにより次元が半分になるので、出力次元を計算
        fc2_in_features = hidden_features // 2
        self.fc2 = nn.Linear(fc2_in_features, in_features)
        
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
