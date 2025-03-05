import torch
import pywt

def apply_wavelet_transform(
    real: torch.Tensor,
    imag: torch.Tensor,
    wavelet_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    How:
        (real, imag) に対して簡単なWavelet変換を適用する例。
        実際には waveletGPT / WavSpA 論文のように複数スケールを扱うが、
        ここでは最小限の離散Wavelet変換(DWT)を適用し、低周波成分を復元して返すなど。

    Why not:
        本来は逆変換など複雑な操作が必要だが、デモとして最低限の実装。

    What:
        shape: (B,S,D)
        wavelet_name 例: "haar", "db1", "db2", ...
    """
    B, S, D = real.shape
    device = real.device

    real_out = torch.zeros_like(real)
    imag_out = torch.zeros_like(imag)

    for b in range(B):
        for d_ in range(D):
            signal_r = real[b, :, d_].cpu().numpy()
            signal_i = imag[b, :, d_].cpu().numpy()

            # 離散Wavelet変換
            cA_r, cD_r = pywt.dwt(signal_r, wavelet_name)
            cA_i, cD_i = pywt.dwt(signal_i, wavelet_name)

            # 簡易的に cA を 2倍に並べて元の長さに合わせる (実際は inverse dwt を使う)
            rec_r, rec_i = [], []
            for idx in range(len(cA_r)):
                rec_r.append(cA_r[idx])
                rec_r.append(cA_r[idx])
                rec_i.append(cA_i[idx])
                rec_i.append(cA_i[idx])
            
            # S を超えないようクリップ
            rec_r = rec_r[:S]
            rec_i = rec_i[:S]

            real_out[b, :, d_] = torch.tensor(rec_r, device=device)
            imag_out[b, :, d_] = torch.tensor(rec_i, device=device)

    return real_out, imag_out