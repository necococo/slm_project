"""
Wave Network モデルの埋め込み分析ツール

波表現の埋め込みを抽出して可視化するための機能を提供します。
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from tqdm import tqdm
from slm.config import PathsConfig
from slm.modules.wave_network import compute_wave_representation


def extract_embeddings(model, dataset, tokenizer, model_config, device, num_samples=100):
    """モデルから波表現の埋め込みを抽出する"""
    print("波表現の埋め込みを抽出しています...")
    model.eval()
    
    # 結果を格納するリスト
    sentence_real_embeds = []
    sentence_imag_embeds = []
    token_real_embeds = []
    token_imag_embeds = []
    
    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Extracting embeddings"):
            # 安全なインデックスアクセス
            idx = i if hasattr(PathsConfig, 'safe_index') else i % len(dataset)
            example = dataset[idx]
            
            # 入力の準備
            input_ids = torch.tensor([example["input_ids"][:model_config.max_seq_len]]).to(device)
            attention_mask = torch.tensor([example["attention_mask"][:model_config.max_seq_len]]).to(device)
            
            # まずトークン埋め込みを取得
            token_embedding = model.token_embedding(input_ids)
            
            # トークンレベル (ローカル) 波表現を計算
            real_part_token, imag_part_token = compute_wave_representation(token_embedding, global_mode=False)
            
            # 文章レベル (グローバル) 波表現を計算
            real_part_sent, imag_part_sent = compute_wave_representation(token_embedding, global_mode=True)
            
            # センテンスレベル埋め込み（シーケンスの平均）
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(real_part_token)
            sentence_real = (real_part_sent * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            sentence_imag = (imag_part_sent * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            
            # 結果を収集
            sentence_real_embeds.append(sentence_real.cpu().numpy())
            sentence_imag_embeds.append(sentence_imag.cpu().numpy())
            token_real_embeds.append(real_part_token[0].cpu().numpy())  # バッチ内の最初の例
            token_imag_embeds.append(imag_part_token[0].cpu().numpy())  # バッチ内の最初の例
    
    # 結果を連結
    sentence_real_embeds = np.concatenate(sentence_real_embeds, axis=0)
    sentence_imag_embeds = np.concatenate(sentence_imag_embeds, axis=0)
    token_real_embeds = np.concatenate(token_real_embeds, axis=0)
    token_imag_embeds = np.concatenate(token_imag_embeds, axis=0)
    
    # 統計情報をログ出力して確認
    print("\n波表現の統計情報:")
    print(f"トークンレベル実部 - 平均値: {np.mean(token_real_embeds):.6f}, 標準偏差: {np.std(token_real_embeds):.6f}")
    print(f"トークンレベル虚部 - 平均値: {np.mean(token_imag_embeds):.6f}, 標準偏差: {np.std(token_imag_embeds):.6f}")
    print(f"センテンスレベル実部 - 平均値: {np.mean(sentence_real_embeds):.6f}, 標準偏差: {np.std(sentence_real_embeds):.6f}")
    print(f"センテンスレベル虚部 - 平均値: {np.mean(sentence_imag_embeds):.6f}, 標準偏差: {np.std(sentence_imag_embeds):.6f}")
    
    return {
        'sentence_real': sentence_real_embeds,
        'sentence_imag': sentence_imag_embeds,
        'token_real': token_real_embeds,
        'token_imag': token_imag_embeds
    }


def visualize_embeddings(embeddings, paths_config):
    """埋め込み表現の分布を可視化します"""
    plt.figure(figsize=(20, 15))
    
    # センテンスレベル実部
    plt.subplot(2, 2, 1)
    plt.hist(embeddings['sentence_real'].flatten(), bins=100, alpha=0.7)
    plt.title('Sentence Level - Real Part Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # センテンスレベル虚部
    plt.subplot(2, 2, 2)
    plt.hist(embeddings['sentence_imag'].flatten(), bins=100, alpha=0.7)
    plt.title('Sentence Level - Imaginary Part Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # トークンレベル実部
    plt.subplot(2, 2, 3)
    plt.hist(embeddings['token_real'].flatten(), bins=100, alpha=0.7)
    plt.title('Token Level - Real Part Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # トークンレベル虚部
    plt.subplot(2, 2, 4)
    plt.hist(embeddings['token_imag'].flatten(), bins=100, alpha=0.7)
    plt.title('Token Level - Imaginary Part Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(paths_config.visualization_path, "embedding_distributions.png")
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    
    # 追加の統計情報
    stats = {}
    for key, value in embeddings.items():
        flat_values = value.flatten()
        stats[key] = {
            'mean': np.mean(flat_values),
            'std': np.std(flat_values),
            'max': np.max(flat_values),
            'min': np.min(flat_values),
            'abs_mean': np.mean(np.abs(flat_values))
        }
    
    # 統計情報の保存
    stats_path = os.path.join(paths_config.visualization_path, "embedding_stats.txt")
    with open(stats_path, 'w') as f:
        f.write("Embedding Distribution Statistics\n")
        f.write("===============================\n\n")
        for key, stat in stats.items():
            f.write(f"{key}:\n")
            for stat_name, stat_value in stat.items():
                f.write(f"  {stat_name}: {stat_value}\n")
            f.write("\n")
    
    print(f"Statistics saved to {stats_path}")
    
    # 追加の可視化：複素平面上の散布図（サンプル）
    plt.figure(figsize=(15, 15))
    
    # センテンスレベル（最初の50要素）
    plt.subplot(1, 2, 1)
    plt.scatter(
        embeddings['sentence_real'][0, :50],
        embeddings['sentence_imag'][0, :50],
        alpha=0.7
    )
    plt.title('Sentence Level - Complex Plane (First 50 dims)')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # トークンレベル（最初のトークンの最初の50次元）
    plt.subplot(1, 2, 2)
    plt.scatter(
        embeddings['token_real'][0, :50],
        embeddings['token_imag'][0, :50],
        alpha=0.7
    )
    plt.title('Token Level - Complex Plane (First 50 dims)')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    scatter_save_path = os.path.join(paths_config.visualization_path, "embedding_scatter.png")
    plt.savefig(scatter_save_path)
    print(f"Scatter plot saved to {scatter_save_path}")


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
