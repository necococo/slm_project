"""
共通ユーティリティ関数
モデル学習と評価の様々なプロセスをサポート
"""
import os
import logging
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Any
import json
import matplotlib.pyplot as plt
from datetime import datetime

def setup_logging(log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """
    ロギング設定をセットアップ
    
    Args:
        log_file: ログファイルのパス（オプション）
        level: ロギングレベル
        
    Returns:
        設定済みのロガー
    """
    # ルートロガー設定
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # フォーマッタ
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # コンソールハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラ（指定されている場合）
    if log_file:
        # ディレクトリ作成（必要な場合）
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def print_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """
    モデルの情報（パラメータ数など）を表示
    
    Args:
        model: 情報を表示するモデル
        
    Returns:
        モデル情報を含む辞書
    """
    info = {}
    
    # パラメータ数の計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info["total_params"] = total_params
    info["total_params_M"] = total_params / 1e6
    info["trainable_params"] = trainable_params
    info["trainable_params_M"] = trainable_params / 1e6
    
    # モデルアーキテクチャの表示
    print(f"Model: {model.__class__.__name__}")
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    return info

def count_model_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    モデルの層ごとのパラメータ数を計算
    
    Args:
        model: パラメータを数えるモデル
        
    Returns:
        レイヤー名とパラメータ数のマッピング
    """
    param_counts = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
            count = sum(p.numel() for p in module.parameters())
            param_counts[name] = count
            
    return param_counts
    
def save_config(config: Any, output_dir: str, filename: str = "config.json") -> str:
    """
    設定をJSONファイルに保存
    
    Args:
        config: 設定オブジェクト
        output_dir: 出力ディレクトリ
        filename: 出力ファイル名
        
    Returns:
        保存したファイルの完全パス
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # 設定内容をdict化
    if hasattr(config, "__dict__"):
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith("_")}
    else:
        config_dict = dict(config)
    
    # datetimeオブジェクトをシリアライズ可能な文字列に変換
    for key, value in config_dict.items():
        if isinstance(value, datetime):
            config_dict[key] = value.isoformat()
    
    # JSONファイルに保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
    return output_path

def plot_training_metrics(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    metrics: Optional[Dict[str, List[float]]] = None,
    output_path: Optional[str] = None
) -> None:
    """
    学習メトリクスをプロット
    
    Args:
        train_losses: 訓練損失のリスト
        val_losses: 検証損失のリスト（オプション）
        metrics: その他のメトリクスの辞書（オプション）
        output_path: 画像保存パス（オプション）
    """
    fig, axes = plt.subplots(nrows=1 + (1 if metrics else 0), ncols=1, figsize=(10, 6 * (1 + (1 if metrics else 0))))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # 損失のプロット
    ax = axes[0]
    epochs = list(range(1, len(train_losses) + 1))
    
    ax.plot(epochs, train_losses, 'b-', label='訓練損失')
    if val_losses:
        ax.plot(epochs, val_losses, 'r-', label='検証損失')
    
    ax.set_title('学習損失')
    ax.set_xlabel('エポック')
    ax.set_ylabel('損失')
    ax.grid(True)
    ax.legend()
    
    # その他のメトリクスをプロット（存在する場合）
    if metrics and len(axes) > 1:
        ax = axes[1]
        
        for name, values in metrics.items():
            ax.plot(epochs[:len(values)], values, label=name)
        
        ax.set_title('学習メトリクス')
        ax.set_xlabel('エポック')
        ax.set_ylabel('値')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    
    # 保存または表示
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
    else:
        plt.show()
    
    plt.close()

def compute_model_flops(model: torch.nn.Module, input_shape: tuple) -> float:
    """
    モデルの推論における浮動小数点演算数を推定
    
    Args:
        model: 計算量を推定するモデル
        input_shape: 入力形状 (batch_size, seq_len)
        
    Returns:
        推定浮動小数点演算数
    """
    # 簡略化されたFLOPs計算
    # 実際には正確な計算にはptflopsなどのライブラリを使用するべき
    batch_size, seq_len = input_shape
    
    if hasattr(model, 'config'):
        hidden_size = getattr(model.config, 'hidden_size', 768)
        num_layers = getattr(model.config, 'num_layers', 12)
        vocab_size = getattr(model.config, 'vocab_size', 30000)
        
        # 埋め込みレイヤーのFLOPs
        embedding_flops = batch_size * seq_len * hidden_size
        
        # アテンションのFLOPs（1層あたり）
        # 簡易計算: QKV射影 + QK注意スコア + 重み付き集約
        attention_flops = batch_size * seq_len * (
            3 * hidden_size * hidden_size +  # QKV射影
            seq_len * hidden_size +          # QK計算
            seq_len * hidden_size            # スコア×V
        )
        
        # フィードフォワードネットワークのFLOPs（1層あたり）
        ffn_flops = batch_size * seq_len * hidden_size * (4 * hidden_size + 4 * hidden_size)
        
        # 層正規化のFLOPs
        norm_flops = batch_size * seq_len * hidden_size * 5  # 簡易推定
        
        # 最終ヘッドのFLOPs
        head_flops = batch_size * seq_len * hidden_size * vocab_size
        
        # 合計
        total_flops = (
            embedding_flops +
            num_layers * (attention_flops + ffn_flops + 2 * norm_flops) +
            norm_flops +
            head_flops
        )
        
        return total_flops
    
    # 設定情報がない場合は簡易推定
    return 0.0

def measure_inference_time(
    model: torch.nn.Module, 
    input_ids: torch.Tensor,
    num_runs: int = 10
) -> float:
    """
    モデル推論時間を測定
    
    Args:
        model: 測定対象のモデル
        input_ids: 入力テンソル
        num_runs: 測定回数
        
    Returns:
        平均推論時間（秒）
    """
    model.eval()
    device = next(model.parameters()).device
    
    if device.type == 'cuda':
        # GPUの場合、初回実行をウォームアップとする
        with torch.no_grad():
            _ = model(input_ids)
        
        # CUDA同期
        torch.cuda.synchronize()
    
    # 推論時間測定
    total_time = 0.0
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_ids)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            total_time += time.time() - start_time
    
    return total_time / num_runs

def memory_stats() -> Dict[str, float]:
    """
    現在のGPUメモリ使用状況を取得
    
    Returns:
        メモリ統計の辞書
    """
    if torch.cuda.is_available():
        stats = {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        return stats
    return {}
