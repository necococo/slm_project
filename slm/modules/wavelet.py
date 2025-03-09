import torch
import torch.nn.functional as F

def apply_haar_wavelet_pytorch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    入力テンソルにHaarウェーブレット変換を適用し、近似成分(approximation)と詳細成分(detail)を返す
    PyTorchのみで実装（numpyを使用しない）
    
    Args:
        x: 入力テンソル [B, S, D]
        
    Returns:
        approx: 低周波成分（近似成分） [B, S, D]
        detail: 高周波成分（詳細成分） [B, S, D]
    """
    # 入力の形状を取得
    B, S, D = x.shape
    device = x.device
    
    # 奇数長の場合は末尾にパディングを追加
    if S % 2 == 1:
        x = F.pad(x, (0, 0, 0, 1, 0, 0), "constant", 0)  # [B, S+1, D]
        padded_S = S + 1
    else:
        padded_S = S
    
    # ローパスフィルタ（スケーリング関数）とハイパスフィルタ（ウェーブレット関数）
    # Haarウェーブレットのフィルタ係数
    low_pass = torch.tensor([0.7071067811865475, 0.7071067811865475], device=device)  # 1/sqrt(2), 1/sqrt(2)
    high_pass = torch.tensor([0.7071067811865475, -0.7071067811865475], device=device)  # 1/sqrt(2), -1/sqrt(2)
    
    # 各バッチと次元について処理
    approx = torch.zeros((B, padded_S, D), device=device)
    detail = torch.zeros((B, padded_S, D), device=device)
    
    # バッチ処理
    for b in range(B):
        for d in range(D):
            # 信号を取得
            signal = x[b, :, d]  # [S] または [S+1]
            
            # 畳み込みと間引きによるダウンサンプリング
            # ステップ1: 信号を2つの部分に分ける（奇数と偶数のインデックス）
            even_indices = torch.arange(0, padded_S, 2, device=device)
            odd_indices = torch.arange(1, padded_S, 2, device=device)
            
            # ステップ2: 各部分に対して適切なフィルタを適用
            even_signal = signal[even_indices]  # [S/2]
            odd_signal = signal[odd_indices]   # [S/2]
            
            # ステップ3: 低周波（近似）と高周波（詳細）成分を計算
            # Haar変換: a_j = (s_{2j} + s_{2j+1})/sqrt(2), d_j = (s_{2j} - s_{2j+1})/sqrt(2)
            cA = (even_signal + odd_signal) * 0.7071067811865475  # [S/2]
            cD = (even_signal - odd_signal) * 0.7071067811865475  # [S/2]
            
            # ステップ4: 結果を元の長さに合わせるための補間（単純な繰り返し）
            # 各値を2回繰り返して元の長さに拡張
            approx_values = torch.repeat_interleave(cA, 2)
            detail_values = torch.repeat_interleave(cD, 2)
            
            # 元の長さにトリミング
            approx[b, :padded_S, d] = approx_values[:padded_S]
            detail[b, :padded_S, d] = detail_values[:padded_S]
    
    # 元の長さに戻す
    if S % 2 == 1:
        approx = approx[:, :S, :]
        detail = detail[:, :S, :]
    
    return approx, detail

def apply_wavelet_transform(
    real: torch.Tensor,
    imag: torch.Tensor,
    wavelet_name: str = "haar"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    実部と虚部の両方にウェーブレット変換を適用し、低周波成分のみを返す
    numpyを使わずにPyTorchのみで実装
    
    Args:
        real: 実部 [B, S, D]
        imag: 虚部 [B, S, D]
        wavelet_name: ウェーブレットの種類（現在はhaarのみ対応）
        
    Returns:
        real_approx: 実部の低周波成分
        imag_approx: 虚部の低周波成分
    """
    if wavelet_name.lower() != "haar":
        raise ValueError("現在はHaarウェーブレットのみ対応しています")
    
    # Haarウェーブレット変換を適用
    real_approx, real_detail = apply_haar_wavelet_pytorch(real)
    imag_approx, imag_detail = apply_haar_wavelet_pytorch(imag)
    
    # 低周波成分のみを返す
    return real_approx, imag_approx

# 元のシグナルから高周波成分を取り出す関数
def extract_high_frequency(x: torch.Tensor, wavelet_name: str = "haar") -> torch.Tensor:
    """
    入力テンソルから高周波成分のみを抽出
    
    Args:
        x: 入力テンソル [B, S, D]
        wavelet_name: ウェーブレットの種類（現在はhaarのみ対応）
        
    Returns:
        detail: 高周波成分 [B, S, D]
    """
    if wavelet_name.lower() != "haar":
        raise ValueError("現在はHaarウェーブレットのみ対応しています")
    
    # Haarウェーブレット変換を適用
    _, detail = apply_haar_wavelet_pytorch(x)
    
    return detail