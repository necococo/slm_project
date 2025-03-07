"""
ウェーブレットモデルの可視化ユーティリティ
アテンションマップ、変換係数、学習曲線などの可視化をサポート
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pywt
from typing import List, Tuple, Optional, Dict, Any, Union

def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    tokens: List[str] = None,
    output_path: Optional[str] = None,
    title: str = "アテンション重み",
    figsize: Tuple[int, int] = (10, 8)
):
    """
    アテンション重みのヒートマップを描画
    
    Args:
        attention_weights: アテンション重みテンソル [seq_len, seq_len]
        tokens: トークン列（表示用）
        output_path: 保存先パス（Noneの場合は表示のみ）
        title: グラフタイトル
        figsize: 図のサイズ
    """
    # テンソルをnumpy配列に変換
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    # トークンがない場合はインデックスを使用
    if tokens is None:
        seq_len = attention_weights.shape[0]
        tokens = [str(i) for i in range(seq_len)]
    
    # ヒートマップの描画
    ax = sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        annot=False,
        fmt='.2f',
        cmap='viridis',
        vmin=0
    )
    
    plt.title(title)
    plt.xlabel('参照トークン')
    plt.ylabel('クエリトークン')
    
    # トークン名が長い場合は傾ける
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 保存または表示
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_wavelet_coefficients(
    coeffs: List[np.ndarray],
    wavelet_name: str = 'db2',
    level: int = 3,
    output_path: Optional[str] = None,
    title: str = "ウェーブレット係数",
    figsize: Tuple[int, int] = (12, 10)
):
    """
    ウェーブレット係数を可視化
    
    Args:
        coeffs: ウェーブレット係数のリスト [cA, cD1, cD2, ...]
        wavelet_name: 使用したウェーブレットの名前
        level: 分解レベル
        output_path: 保存先パス（Noneの場合は表示のみ）
        title: グラフタイトル
        figsize: 図のサイズ
    """
    # テンソルをnumpy配列に変換
    if isinstance(coeffs[0], torch.Tensor):
        coeffs = [c.detach().cpu().numpy() for c in coeffs]
    
    plt.figure(figsize=figsize)
    
    # サブプロット数: 近似係数 + 詳細係数
    n_subplots = level + 1
    
    # 近似係数のプロット
    plt.subplot(n_subplots, 1, 1)
    plt.plot(coeffs[0])
    plt.title(f"{title} - 近似係数 ({wavelet_name})")
    plt.ylabel("振幅")
    plt.grid(True, alpha=0.3)
    
    # 詳細係数のプロット（各レベル）
    for i in range(level):
        plt.subplot(n_subplots, 1, i + 2)
        plt.plot(coeffs[i+1])
        plt.title(f"{title} - 詳細係数 レベル {i+1}")
        plt.ylabel("振幅")
        plt.grid(True, alpha=0.3)
    
    plt.xlabel("サンプル位置")
    plt.tight_layout()
    
    # 保存または表示
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_wavelet_decomposition(
    signal: np.ndarray,
    wavelet_name: str = 'db2',
    level: int = 3,
    output_path: Optional[str] = None,
    title: str = "信号とウェーブレット分解",
    figsize: Tuple[int, int] = (12, 12)
):
    """
    信号のウェーブレット分解を視覚化
    
    Args:
        signal: 入力信号
        wavelet_name: ウェーブレット名
        level: 分解レベル
        output_path: 保存先パス（Noneの場合は表示のみ）
        title: グラフタイトル
        figsize: 図のサイズ
    """
    # テンソルをnumpy配列に変換
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()
    
    # 1次元信号に変換
    if signal.ndim > 1:
        signal = signal.flatten()
    
    # ウェーブレット分解
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    
    # サブプロット作成
    fig, axes = plt.subplots(level + 2, 1, figsize=figsize)
    
    # 元の信号
    axes[0].plot(signal)
    axes[0].set_title(f"{title} - 元信号")
    axes[0].set_ylabel("振幅")
    axes[0].grid(True, alpha=0.3)
    
    # 近似係数
    axes[1].plot(coeffs[0])
    axes[1].set_title(f"近似係数 ({wavelet_name})")
    axes[1].set_ylabel("振幅")
    axes[1].grid(True, alpha=0.3)
    
    # 詳細係数（各レベル）
    for i in range(level):
        axes[i+2].plot(coeffs[i+1])
        axes[i+2].set_title(f"詳細係数 レベル {i+1}")
        axes[i+2].set_ylabel("振幅")
        axes[i+2].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("サンプル位置")
    plt.tight_layout()
    
    # 保存または表示
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    学習履歴をプロット
    
    Args:
        history: 学習履歴の辞書 {'train_loss': [...], 'val_loss': [...], ...}
        output_path: 保存先パス（Noneの場合は表示のみ）
        figsize: 図のサイズ
    """
    plt.figure(figsize=figsize)
    
    # 損失のプロット
    if 'train_loss' in history:
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], 'b-', marker='o', label='訓練損失')
    
    if 'val_loss' in history:
        epochs = range(1, len(history['val_loss']) + 1)
        plt.plot(epochs, history['val_loss'], 'r-', marker='s', label='検証損失')
    
    # その他のメトリクスのプロット
    for key, values in history.items():
        if key not in ['train_loss', 'val_loss']:
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, marker='d', label=key)
    
    plt.title('学習履歴')
    plt.xlabel('エポック')
    plt.ylabel('損失/メトリクス')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 保存または表示
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def compare_models(
    model_results: Dict[str, Dict[str, List[float]]],
    metric: str = 'val_loss',
    output_path: Optional[str] = None,
    title: str = "モデル比較",
    figsize: Tuple[int, int] = (12, 8)
):
    """
    複数のモデルのパフォーマンスを比較
    
    Args:
        model_results: モデルごとの結果辞書 {'model_a': {'val_loss': [...]}, ...}
        metric: 比較するメトリクス
        output_path: 保存先パス（Noneの場合は表示のみ）
        title: グラフタイトル
        figsize: 図のサイズ
    """
    plt.figure(figsize=figsize)
    
    for model_name, results in model_results.items():
        if metric in results:
            epochs = range(1, len(results[metric]) + 1)
            plt.plot(epochs, results[metric], marker='o', label=model_name)
    
    plt.title(title)
    plt.xlabel('エポック')
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 保存または表示
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_attention_pattern_comparison(
    attention_maps: Dict[str, np.ndarray],
    tokens: List[str] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    異なるタイプのアテンションパターンを比較
    
    Args:
        attention_maps: アテンションマップの辞書 {'svd': np.ndarray, 'linear': np.ndarray, ...}
        tokens: トークンリスト（表示用）
        output_path: 保存先パス（Noneの場合は表示のみ）
        figsize: 図のサイズ
    """
    n_maps = len(attention_maps)
    fig, axes = plt.subplots(1, n_maps, figsize=figsize)
    
    # トークンがない場合は空リストを使用
    if tokens is None:
        first_map = list(attention_maps.values())[0]
        seq_len = first_map.shape[0]
        tokens = [str(i) for i in range(seq_len)]
    
    # 各アテンションマップをプロット
    for i, (name, attn_map) in enumerate(attention_maps.items()):
        if n_maps == 1:
            ax = axes
        else:
            ax = axes[i]
        
        # テンソルをnumpy配列に変換
        if isinstance(attn_map, torch.Tensor):
            attn_map = attn_map.detach().cpu().numpy()
        
        # ヒートマップの描画
        sns.heatmap(
            attn_map,
            xticklabels=tokens,
            yticklabels=tokens,
            annot=False,
            cmap='viridis',
            ax=ax,
            vmin=0
        )
        
        ax.set_title(f"{name}アテンション")
        ax.set_xlabel('参照トークン')
        ax.set_ylabel('クエリトークン')
        
        # トークン名が長い場合は傾ける
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 保存または表示
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()
