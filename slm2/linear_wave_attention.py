"""
線形時間計算量の波動表現アテンション機構
通常のアテンション計算のO(n²)の複雑性を回避し、O(n)で効率的に計算します
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Union

class LinearWaveAttention(nn.Module):
    """
    波動表現に特化した線形計算量のアテンション機構
    
    通常のセルフアテンションの計算量O(n²)から、
    特殊な特徴マッピングにより計算量O(n)を実現します。
    
    参考文献:
    - Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
    - Linear Transformers Are Secretly Fast Weight Programmers
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        kernel_fn: str = "elu",  # "elu", "relu", "squared_relu"
        eps: float = 1e-6,
        causal: bool = False,
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            num_heads: アテンションヘッドの数
            head_dim: 各ヘッドの次元 (指定なしの場合はhidden_size // num_heads)
            dropout: ドロップアウト率
            kernel_fn: 使用するカーネル関数
            eps: 数値安定性のための小さな値
            causal: 因果的マスクを適用するかどうか
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        if self.head_dim * num_heads != hidden_size:
            # 次元数の調整（総次元数を維持）
            self.head_dim = math.ceil(hidden_size / num_heads)
            self.num_heads = math.floor(hidden_size / self.head_dim)
            print(f"警告: hidden_sizeがnum_headsで割り切れません。"
                  f"num_heads={self.num_heads}, head_dim={self.head_dim}に調整")
        
        self.total_head_dim = self.num_heads * self.head_dim
        self.eps = eps
        self.kernel_fn = kernel_fn
        self.causal = causal
        
        # クエリ・キー・バリューの射影
        self.q_proj = nn.Linear(hidden_size, self.total_head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.total_head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.total_head_dim, bias=False)
        
        # 出力射影
        self.output_proj = nn.Linear(self.total_head_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # 波動表現用の複素型変換
        self.register_buffer('Complex_i', torch.tensor([0.0, 1.0]))  # 虚数単位 i
        
    def _apply_kernel_fn(self, x: torch.Tensor) -> torch.Tensor:
        """
        特徴マップ関数を適用
        
        Args:
            x: 入力テンソル [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            特徴マップ適用後のテンソル
        """
        if self.kernel_fn == "elu":
            return F.elu(x) + 1.0
        elif self.kernel_fn == "relu":
            return F.relu(x)
        elif self.kernel_fn == "squared_relu":
            return F.relu(x) ** 2
        else:
            raise ValueError(f"不明なカーネル関数: {self.kernel_fn}")
    
    def _wave_feature_map(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        """
        波動表現に特化した特徴マップを適用
        
        Args:
            real: 実部 [batch_size, num_heads, seq_len, head_dim]
            imag: 虚部 [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            特徴マップ適用後のテンソル
        """
        # 複素数の絶対値（振幅）
        amplitude = torch.sqrt(real ** 2 + imag ** 2 + self.eps)
        
        # 位相（-π からπ）
        phase = torch.atan2(imag, real)
        
        # 位相の正規化（複素平面上のスムースな写像）
        normalized_phase = torch.sin(phase)
        
        # 非線形変換を適用した振幅と位相を組み合わせる
        return self._apply_kernel_fn(amplitude) * torch.exp(normalized_phase * self.Complex_i.to(amplitude.device))
    
    def forward(self, wave_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        線形波動注意機構のフォワードパス
        
        Args:
            wave_hidden_states: 波動表現の隠れ状態 [batch_size, seq_len, hidden_size*2]
                                最初の半分が実部、後半が虚部
            
        Returns:
            処理済みテンソル [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size_double = wave_hidden_states.shape
        assert hidden_size_double == self.hidden_size * 2, f"入力次元が想定と異なります: {hidden_size_double} vs {self.hidden_size * 2}"
        
        # 実部と虚部を分離
        hidden_size = self.hidden_size
        real = wave_hidden_states[:, :, :hidden_size]  # 前半
        imag = wave_hidden_states[:, :, hidden_size:]  # 後半
        
        # QKV射影（実部と虚部を別々に処理）
        q_real = self.q_proj(real)  # [B, S, H*D]
        k_real = self.k_proj(real)
        v_real = self.v_proj(real)
        
        q_imag = self.q_proj(imag)
        k_imag = self.k_proj(imag)
        v_imag = self.v_proj(imag)
        
        # 多頭表現に変形
        q_real = q_real.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, S, D]
        k_real = k_real.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_real = v_real.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        q_imag = q_imag.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_imag = k_imag.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_imag = v_imag.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 波動特徴マッピング
        q = self._wave_feature_map(q_real, q_imag)
        k = self._wave_feature_map(k_real, k_imag)
        
        # 実部のみ抽出（キーと値の内積計算のため）
        q = q.real
        k = k.real
        
        # 線形アテンション計算（行列乗算の順序を変更してO(n)に）
        if self.causal:
            # 因果的マスク付きの線形アテンション（自己回帰モデル向け）
            output = self._causal_linear_attention(q, k, v)
        else:
            # 標準的な線形アテンション
            kv = torch.einsum('bhnd,bhne->bhde', k, v)  # [B, H, D, E]
            output = torch.einsum('bhsd,bhde->bhse', q, kv)  # [B, H, S, E]
        
        # 結合と出力射影
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)  # [B, S, H*D]
        output = self.output_proj(output)  # [B, S, hidden_size]
        output = self.dropout(output)
        
        return output
    
    def _causal_linear_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        因果的線形アテンションの計算（自己回帰生成向け）
        
        Args:
            q: クエリ [B, H, S, D]
            k: キー [B, H, S, D]
            v: 値 [B, H, S, D]
            
        Returns:
            アテンション出力 [B, H, S, D]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # 累積和を使った効率的な因果的アテンション
        output = torch.zeros_like(v)
        
        # 位置ごとに処理
        cumulative_k = torch.zeros(batch_size, num_heads, head_dim, head_dim, device=k.device)
        cumulative_v = torch.zeros(batch_size, num_heads, head_dim, device=v.device)
        
        for t in range(seq_len):
            # 現在位置までのキーと値の累積
            cumulative_k = cumulative_k + torch.einsum('bhd,bhe->bhde', k[:, :, t], v[:, :, t])
            
            # 現在のクエリと累積キー・値の計算
            output[:, :, t] = torch.einsum('bhd,bhde->bhe', q[:, :, t], cumulative_k)
        
        return output


class WaveNormalization(nn.Module):
    """
    波動表現に特化した正規化レイヤー
    
    振幅と位相の両方を正しく処理する特殊な正規化
    """
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            eps: 数値安定性のための小さな値
            elementwise_affine: アフィン変換を学習するか
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight_real = nn.Parameter(torch.ones(hidden_size))
            self.weight_imag = nn.Parameter(torch.ones(hidden_size))
            self.bias_real = nn.Parameter(torch.zeros(hidden_size))
            self.bias_imag = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        波動表現の正規化
        
        Args:
            real: 実部 [batch_size, seq_len, hidden_size]
            imag: 虚部 [batch_size, seq_len, hidden_size]
            
        Returns:
            正規化された (実部, 虚部) のタプル
        """
        # 振幅の計算
        amplitude = torch.sqrt(real ** 2 + imag ** 2 + self.eps)
        
        # 平均振幅
        mean_amplitude = amplitude.mean(dim=-1, keepdim=True)
        
        # 振幅の標準偏差
        std_amplitude = amplitude.std(dim=-1, keepdim=True) + self.eps
        
        # 正規化
        real_norm = real / (mean_amplitude * std_amplitude)
        imag_norm = imag / (mean_amplitude * std_amplitude)
        
        # アフィン変換（オプション）
        if self.elementwise_affine:
            real_norm = real_norm * self.weight_real + self.bias_real
            imag_norm = imag_norm * self.weight_imag + self.bias_imag
        
        return real_norm, imag_norm


class LinformerWaveAttention(nn.Module):
    """
    Linformer（低ランク近似）と波動表現を組み合わせた
    効率的な線形時間アテンション機構
    
    Linformerの低ランク行列とWave表現による
    さらなる効率化が特徴
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        projection_dim: int = 256,  # 低ランク投影次元
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            num_heads: アテンションヘッドの数
            head_dim: 各ヘッドの次元 (指定なしの場合はhidden_size // num_heads)
            dropout: ドロップアウト率
            max_seq_len: 最大シーケンス長
            projection_dim: シーケンス長を射影する次元（低ランク近似の次元）
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.max_seq_len = max_seq_len
        self.projection_dim = projection_dim
        
        self.total_head_dim = self.num_heads * self.head_dim
        
        # クエリ・キー・バリューの射影
        self.q_proj = nn.Linear(hidden_size, self.total_head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.total_head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.total_head_dim, bias=False)
        
        # Linformerの低ランク射影行列 E, F
        self.e_proj = nn.Parameter(torch.Tensor(self.num_heads, projection_dim, max_seq_len))
        self.f_proj = nn.Parameter(torch.Tensor(self.num_heads, projection_dim, max_seq_len))
        
        # 出力射影
        self.output_proj = nn.Linear(self.total_head_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # 初期化
        nn.init.xavier_uniform_(self.e_proj)
        nn.init.xavier_uniform_(self.f_proj)
        
    def forward(self, wave_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Linformer波動注意機構のフォワードパス
        
        Args:
            wave_hidden_states: 波動表現の隠れ状態 [batch_size, seq_len, hidden_size*2]
                                最初の半分が実部、後半が虚部
            
        Returns:
            処理済みテンソル [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size_double = wave_hidden_states.shape
        hidden_size = self.hidden_size
        
        # 実部と虚部を分離
        real = wave_hidden_states[:, :, :hidden_size]  # 前半
        imag = wave_hidden_states[:, :, hidden_size:]  # 後半
        
        # 実部と虚部を組み合わせた波動振幅を計算
        amplitude = torch.sqrt(real ** 2 + imag ** 2 + 1e-6)
        
        # 量子位相の計算
        phase = torch.atan2(imag, real)
        
        # QKV射影
        q = self.q_proj(amplitude)  # [B, S, H*D]
        k = self.k_proj(amplitude * torch.cos(phase))
        v = self.v_proj(amplitude * torch.sin(phase))
        
        # 多頭表現に変形
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, S, D]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # キーと値の低ランク射影（シーケンス長次元の削減）
        k_proj = torch.matmul(self.e_proj[:, :, :seq_len], k)  # [B, H, P, D]
        v_proj = torch.matmul(self.f_proj[:, :, :seq_len], v)  # [B, H, P, D]
        
        # スケーリング
        scaling = float(self.head_dim) ** -0.5
        q = q * scaling
        
        # QK^TV計算（低ランク近似により効率化）
        attn_scores = torch.matmul(q, k_proj.transpose(-1, -2))  # [B, H, S, P]
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 値との積
        output = torch.matmul(attn_probs, v_proj)  # [B, H, S, D]
        
        # 結合と出力射影
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)  # [B, S, H*D]
        output = self.output_proj(output)  # [B, S, hidden_size]
        output = self.dropout(output)
        
        return output
