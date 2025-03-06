import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        norm = x.norm(dim=-1, keepdim=True) / (x.size(-1) ** 0.5)
        x_norm = x / (norm + self.eps)
        return self.weight * x_norm
