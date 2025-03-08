"""
Wave Networkの波表現（複素数表現）における分布パターンの詳細分析ツール
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def analyze_wave_embeddings(embeddings, save_dir):
    """波表現の詳細な分析と可視化を行います"""
    # より詳細なヒストグラムと分布プロット
    plt.figure(figsize=(18, 12))
    
    # カラーパレット設定
    sns.set_palette("viridis")
    
    # センテンスレベル実部と虚部の分布比較（KDEプロット）
    plt.subplot(2, 2, 1)
    sns.kdeplot(embeddings['sentence_real'].flatten(), label='Real Part')
    sns.kdeplot(embeddings['sentence_imag'].flatten(), label='Imaginary Part')
    plt.title('Sentence Level - Distribution Comparison')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # トークンレベル実部と虚部の分布比較（KDEプロット）
    plt.subplot(2, 2, 2)
    sns.kdeplot(embeddings['token_real'].flatten(), label='Real Part')
    sns.kdeplot(embeddings['token_imag'].flatten(), label='Imaginary Part')
    plt.title('Token Level - Distribution Comparison')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # センテンスレベル虚部（正規分布との比較）
    plt.subplot(2, 2, 3)
    imag_data = embeddings['sentence_imag'].flatten()
    sns.histplot(imag_data, kde=True, stat="density", label="Actual")
    x = np.linspace(imag_data.min(), imag_data.max(), 100)
    plt.plot(x, np.exp(-(x-np.mean(imag_data))**2/(2*np.var(imag_data))) / 
             np.sqrt(2*np.pi*np.var(imag_data)),
             'r-', label='Normal Distribution')
    plt.title('Sentence Imaginary Part vs Normal Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 複素平面上の分布（サンプルポイント）
    plt.subplot(2, 2, 4)
    # 各埋め込みから500ポイントをランダムサンプリング
    idx = np.random.choice(embeddings['sentence_real'].shape[0] * embeddings['sentence_real'].shape[1], 
                          size=500, replace=False)
    real_flat = embeddings['sentence_real'].flatten()[idx]
    imag_flat = embeddings['sentence_imag'].flatten()[idx]
    plt.scatter(real_flat, imag_flat, alpha=0.6, s=30)
    plt.title('Complex Plane Distribution (500 Sample Points)')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(save_dir, "wave_distribution_analysis.png")
    plt.savefig(save_path)
    print(f"詳細な分布分析を保存しました: {save_path}")
    
    # 追加の分析統計
    print("\n=== 波表現の詳細統計 ===")
    print("センテンスレベル虚部の歪度:", scipy.stats.skew(embeddings['sentence_imag'].flatten()))
    print("センテンスレベル虚部の尖度:", scipy.stats.kurtosis(embeddings['sentence_imag'].flatten()))
    print("トークンレベル虚部の歪度:", scipy.stats.skew(embeddings['token_imag'].flatten()))
    print("トークンレベル虚部の尖度:", scipy.stats.kurtosis(embeddings['token_imag'].flatten()))
    
    return save_path
