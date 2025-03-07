"""
Wavelet変換を行うレイヤー
ニューラルネットワークにおけるWavelet変換の実装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union

class WaveletLayer(nn.Module):
    """
    標準的なWavelet変換レイヤー
    
    入力テンソルに対してWavelet変換を適用し、
    マルチスケール特徴を抽出する
    """
    def __init__(
        self,
        hidden_size: int,
        wavelet_name: str = "db2",
        level: int = 1,
        mode: str = "symmetric",
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            wavelet_name: 使用するウェーブレット (pywt互換の名前)
            level: 分解レベル
            mode: 境界の扱い方 (symmetric, periodic など)
            dropout: ドロップアウト率
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.wavelet_name = wavelet_name
        self.level = level
        self.mode = mode
        
        # Wavelet変換後の係数を処理するための畳み込み層
        self.conv1d_approx = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv1d_detail = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        
        # 出力投影
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def _apply_wavelet(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        トーチテンソルにWavelet変換を適用
        
        Args:
            x: 入力テンソル [batch_size, seq_len, hidden_size]
            
        Returns:
            Wavelet係数のリスト [approx, detail]
        """
        # 形状を取得
        batch_size, seq_len, hidden_size = x.shape
        device = x.device
        dtype = x.dtype
        
        # CPUに移動（PyWaveletsはCPUのみ対応）
        x_cpu = x.detach().cpu().numpy()
        
        # 特徴量ごとに処理
        coeffs_approx = np.zeros((batch_size, seq_len // 2, hidden_size))
        coeffs_detail = np.zeros((batch_size, seq_len // 2, hidden_size))
        
        # バッチとチャンネル方向にループ
        for b in range(batch_size):
            for h in range(hidden_size):
                # 1次元配列に対してwavelet分解
                signal = x_cpu[b, :, h]
                coeffs = pywt.wavedec(signal, self.wavelet_name, mode=self.mode, level=self.level)
                
                # 近似係数と詳細係数を抽出
                approx, detail = coeffs[0], coeffs[1]
                
                # サイズが異なる場合は調整（パディングやトリミング）
                target_len = seq_len // 2
                if len(approx) < target_len:
                    pad_len = target_len - len(approx)
                    coeffs_approx[b, :, h] = np.pad(approx, (0, pad_len), mode='constant')
                    coeffs_detail[b, :, h] = np.pad(detail, (0, pad_len), mode='constant')
                else:
                    coeffs_approx[b, :, h] = approx[:target_len]
                    coeffs_detail[b, :, h] = detail[:target_len]
        
        # テンソルに戻す
        coeffs_approx_tensor = torch.tensor(coeffs_approx, device=device, dtype=dtype)
        coeffs_detail_tensor = torch.tensor(coeffs_detail, device=device, dtype=dtype)
        
        return [coeffs_approx_tensor, coeffs_detail_tensor]
    
    def _inverse_wavelet(self, coeffs: List[torch.Tensor], orig_size: int) -> torch.Tensor:
        """
        Wavelet係数から信号を再構築
        
        Args:
            coeffs: Wavelet係数のリスト [approx, detail, ...]
            orig_size: 元の信号長
            
        Returns:
            再構築された信号
        """
        # 形状を取得
        batch_size, reduced_len, hidden_size = coeffs[0].shape
        device = coeffs[0].device
        dtype = coeffs[0].dtype
        
        # CPUに移動
        coeffs_cpu = [c.detach().cpu().numpy() for c in coeffs]
        
        # 結果を格納する配列
        result = np.zeros((batch_size, orig_size, hidden_size))
        
        # バッチとチャンネル方向にループ
        for b in range(batch_size):
            for h in range(hidden_size):
                # 各特徴の係数を抽出
                approx = coeffs_cpu[0][b, :, h]
                detail = coeffs_cpu[1][b, :, h]
                
                # 逆変換用のリスト作成
                coeff_list = [approx, detail] + [None] * (self.level - 1)
                
                # 逆変換
                reconstructed = pywt.waverec(coeff_list, self.wavelet_name, mode=self.mode)
                
                # 元のサイズに切り詰め
                if len(reconstructed) >= orig_size:
                    result[b, :, h] = reconstructed[:orig_size]
                else:
                    result[b, :, h] = np.pad(reconstructed, (0, orig_size - len(reconstructed)))
        
        # テンソルに戻す
        return torch.tensor(result, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        WaveletLayerのフォワードパス
        
        Args:
            x: 入力テンソル [batch_size, seq_len, hidden_size]
            
        Returns:
            処理後のテンソル [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # 入力のシーケンス長が偶数でない場合はパディング
        if seq_len % 2 != 0:
            pad = torch.zeros(batch_size, 1, hidden_size, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
            seq_len += 1
            
        # Wavelet変換を適用
        coeffs = self._apply_wavelet(x)
        approx, detail = coeffs[0], coeffs[1]
        
        # チャンネル次元を前に持ってくる（畳み込み用）
        approx_conv = self.conv1d_approx(approx.transpose(1, 2)).transpose(1, 2)
        detail_conv = self.conv1d_detail(detail.transpose(1, 2)).transpose(1, 2)
        
        # 係数を処理
        processed_approx = F.gelu(approx_conv)
        processed_detail = F.gelu(detail_conv)
        
        # 係数を結合して逆変換
        processed_coeffs = [processed_approx, processed_detail]
        reconstructed = self._inverse_wavelet(processed_coeffs, seq_len)
        
        # 出力射影
        output = self.output_proj(reconstructed)
        output = self.dropout(output)
        
        # パディングした場合は元のサイズに戻す
        if seq_len != x.shape[1]:
            output = output[:, :x.shape[1], :]
            
        return output


class FastWaveletLayer(nn.Module):
    """
    高速Wavelet変換レイヤー
    
    PyTorchの畳み込みを使用して近似的なWavelet変換を実行し、
    GPUでの高速処理を可能にします
    """
    def __init__(
        self,
        hidden_size: int,
        wavelet_name: str = "db2",
        level: int = 1,
        dropout: float = 0.1,
        use_separate_params: bool = True
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            wavelet_name: 使用するウェーブレット (参考用)
            level: 分解レベル
            dropout: ドロップアウト率
            use_separate_params: 特徴次元ごとに別パラメータを使うか
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.wavelet_name = wavelet_name
        self.level = level
        self.use_separate_params = use_separate_params
        
        # Waveletフィルタ係数の初期化
        # db2ウェーブレットのフィルタ係数に近いものを使用
        db2_approx = [0.3415, 0.5915, 0.1585, -0.0915]
        db2_detail = [-0.0915, -0.1585, 0.5915, -0.3415]
        
        filter_length = len(db2_approx)
        pad_size = (filter_length - 1) // 2
        
        if use_separate_params:
            # 特徴量ごとに別々のフィルタ
            self.approx_conv = nn.Conv1d(
                hidden_size, hidden_size, 
                kernel_size=filter_length, 
                stride=2, 
                padding=pad_size,
                groups=hidden_size,  # 深さ方向の畳み込み
                bias=False
            )
            
            self.detail_conv = nn.Conv1d(
                hidden_size, hidden_size, 
                kernel_size=filter_length, 
                stride=2, 
                padding=pad_size,
                groups=hidden_size,  # 深さ方向の畳み込み
                bias=False
            )
            
            # フィルタの初期化
            with torch.no_grad():
                for i in range(hidden_size):
                    self.approx_conv.weight[i, 0, :] = torch.tensor(db2_approx)
                    self.detail_conv.weight[i, 0, :] = torch.tensor(db2_detail)
        else:
            # すべての特徴量で同じフィルタ
            self.register_buffer('approx_filter', 
                                torch.tensor(db2_approx).view(1, 1, filter_length))
            self.register_buffer('detail_filter', 
                                torch.tensor(db2_detail).view(1, 1, filter_length))
        
        # 処理用の層
        self.approx_proj = nn.Linear(hidden_size, hidden_size)
        self.detail_proj = nn.Linear(hidden_size, hidden_size)
        
        # 出力投影
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        高速Wavelet変換レイヤーのフォワードパス
        
        Args:
            x: 入力テンソル [batch_size, seq_len, hidden_size]
            
        Returns:
            処理後のテンソル [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # 入力のシーケンス長が偶数でない場合はパディング
        orig_seq_len = seq_len
        if seq_len % 2 != 0:
            pad = torch.zeros(batch_size, 1, hidden_size, device=x.device)
            x = torch.cat([x, pad], dim=1)
            seq_len += 1
            
        # チャンネル次元を前に持ってくる
        x_transposed = x.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
        
        # Wavelet変換（畳み込みで近似）
        if self.use_separate_params:
            approx = self.approx_conv(x_transposed)  # [batch_size, hidden_size, seq_len/2]
            detail = self.detail_conv(x_transposed)
        else:
            # 各特徴次元に同じフィルタを適用
            approx = F.conv1d(
                x_transposed, 
                self.approx_filter.expand(hidden_size, 1, -1), 
                stride=2, 
                padding=(self.approx_filter.shape[2] - 1) // 2,
                groups=hidden_size
            )
            
            detail = F.conv1d(
                x_transposed, 
                self.detail_filter.expand(hidden_size, 1, -1), 
                stride=2, 
                padding=(self.detail_filter.shape[2] - 1) // 2,
                groups=hidden_size
            )
        
        # チャンネル次元を戻す
        approx = approx.transpose(1, 2)  # [batch_size, seq_len/2, hidden_size]
        detail = detail.transpose(1, 2)
        
        # 係数の処理
        processed_approx = self.approx_proj(approx)
        processed_detail = self.detail_proj(detail)
        
        # 簡易的な逆変換（アップサンプリングと同等）
        upsampled_approx = F.interpolate(
            processed_approx.transpose(1, 2), 
            size=seq_len, 
            mode='linear', 
            align_corners=False
        ).transpose(1, 2)
        
        upsampled_detail = F.interpolate(
            processed_detail.transpose(1, 2), 
            size=seq_len, 
            mode='linear', 
            align_corners=False
        ).transpose(1, 2)
        
        # 信号再構築
        reconstructed = upsampled_approx + upsampled_detail
        
        # 出力射影
        output = self.output_proj(reconstructed)
        output = self.dropout(output)
        
        # パディングした場合は元のサイズに戻す
        if orig_seq_len != seq_len:
            output = output[:, :orig_seq_len, :]
            
        return output


class MultiResolutionWaveletLayer(nn.Module):
    """
    複数のウェーブレットを同時に使用するマルチレゾリューション層
    
    異なる種類のウェーブレットを組み合わせることで
    多様なスケールと特性の特徴を抽出
    """
    def __init__(
        self,
        hidden_size: int,
        wavelet_families: List[str] = ["db2", "haar", "sym2"],
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            wavelet_families: 使用するウェーブレットの種類のリスト
            dropout: ドロップアウト率
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.wavelet_families = wavelet_families
        
        # 各ウェーブレットに対応するレイヤー
        self.wavelet_layers = nn.ModuleList([
            FastWaveletLayer(
                hidden_size=hidden_size // len(wavelet_families),
                wavelet_name=wavelet_name,
                dropout=dropout
            )
            for wavelet_name in wavelet_families
        ])
        
        # 出力融合
        self.fusion = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        マルチレゾリューションWaveletのフォワードパス
        
        Args:
            x: 入力テンソル [batch_size, seq_len, hidden_size]
            
        Returns:
            処理後のテンソル [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # 特徴次元を均等に分割
        split_size = self.hidden_size // len(self.wavelet_families)
        splits = torch.split(x, split_size, dim=-1)
        
        # 各ウェーブレットレイヤーを適用
        wavelet_outputs = []
        for i, layer in enumerate(self.wavelet_layers):
            output = layer(splits[i])
            wavelet_outputs.append(output)
        
        # 出力を結合
        combined = torch.cat(wavelet_outputs, dim=-1)
        
        # 出力融合
        output = self.fusion(combined)
        output = self.dropout(output)
        
        return output


class WaveletAttention(nn.Module):
    """
    ウェーブレット変換と注意機構を組み合わせた効率的な層
    
    計算量を削減しつつ、トークン間の長距離依存関係を捉えます。
    """
    def __init__(
        self,
        hidden_size: int,
        wavelet_family: str = "db4",
        num_heads: int = 8,
        dropout: float = 0.1,
        use_linear_attention: bool = True  # 線形計算量の注意機構を使うかどうか
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            wavelet_family: 使用するウェーブレット関数
            num_heads: アテンションヘッド数
            dropout: ドロップアウト率
            use_linear_attention: 線形計算量の注意機構を使うかどうか
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_linear_attention = use_linear_attention
        
        # クエリ・キー・バリュー射影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # ウェーブレット処理
        self.wavelet_layer = WaveletLayer(
            hidden_size=hidden_size,
            wavelet_family=wavelet_family,
            decomposition_level=2,
            dropout=dropout
        )
        
        # 出力投影
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ウェーブレットアテンションのフォワードパス
        
        Args:
            x: 入力テンソル [batch_size, seq_len, hidden_size]
            
        Returns:
            処理後のテンソル
        """
        batch_size, seq_len, _ = x.shape
        
        # 残差接続用に保存
        identity = x
        
        # クエリ・キー・バリュー射影
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 形状変換: [batch_size, num_heads, seq_len, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # 線形アテンションを使う場合
        if self.use_linear_attention:
            # FFTを使った線形計算量の注意計算
            # Q・K^T・V ではなく FFT(Q) * FFT(K)^* * FFT(V) を計算
            q_fft = torch.fft.rfft(q, dim=2)
            k_fft = torch.fft.rfft(k, dim=2)
            v_fft = torch.fft.rfft(v, dim=2)
            
            # 周波数領域での乗算
            output_fft = q_fft * k_fft.conj() * v_fft
            
            # 逆変換
            output = torch.fft.irfft(output_fft, n=seq_len, dim=2)
            
            # スケーリング
            output = output / math.sqrt(seq_len)
        else:
            # 通常の注意計算 (O(n²))
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_weights, dim=-1)
            output = torch.matmul(attn_weights, v)
        
        # 形状を戻す: [batch_size, seq_len, hidden_size]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # ウェーブレット変換を適用
        output = self.wavelet_layer(output)
        
        # 出力投影
        output = self.output_proj(output)
        output = self.dropout(output)
        
        # 残差接続とレイヤー正規化
        output = self.layer_norm(output + identity)
        
        return output
