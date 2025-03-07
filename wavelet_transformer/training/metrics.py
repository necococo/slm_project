"""
モデル評価のための指標
MLMタスクとQAタスクの両方の指標を提供
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import math

def compute_perplexity(loss: float) -> float:
    """
    交差エントロピー損失からperplexityを計算
    
    Args:
        loss: 交差エントロピー損失
        
    Returns:
        perplexity値
    """
    return float(np.exp(min(loss, 30)))  # 数値安定性のため上限を設定

def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """
    予測精度を計算
    
    Args:
        predictions: モデルの予測 [batch_size, seq_len, vocab_size] or [batch_size, seq_len]
        labels: ターゲットラベル [batch_size, seq_len]
        ignore_index: 無視するラベルインデックス
        
    Returns:
        予測精度 (0.0-1.0)
    """
    if predictions.dim() > 2:
        predictions = predictions.argmax(dim=-1)
    
    # 有効な位置のマスクを作成
    mask = (labels != ignore_index)
    
    # マスク位置がない場合は0を返す
    if mask.sum() == 0:
        return 0.0
    
    # 正解とマスクされた位置だけを比較
    correct = ((predictions == labels) & mask).sum().item()
    total = mask.sum().item()
    
    return correct / total

def compute_mlm_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    MLMタスクの評価指標を計算
    
    Args:
        logits: モデルの予測ロジット [batch_size, seq_len, vocab_size]
        labels: ターゲットラベル [batch_size, seq_len]
        loss: 計算済み損失（オプション）
        
    Returns:
        評価指標の辞書
    """
    predictions = logits.argmax(dim=-1)
    accuracy = compute_accuracy(predictions, labels)
    
    # 損失が計算されていれば、perplexityを計算
    perplexity = None
    if loss is not None:
        perplexity = compute_perplexity(loss.item())
        
    metrics = {
        "accuracy": accuracy
    }
    
    if perplexity is not None:
        metrics["perplexity"] = perplexity
        
    return metrics

def compute_qa_metrics(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    start_positions: torch.Tensor,
    end_positions: torch.Tensor
) -> Dict[str, float]:
    """
    QAタスクの評価指標を計算
    
    Args:
        start_logits: 開始位置の予測 [batch_size, seq_len]
        end_logits: 終了位置の予測 [batch_size, seq_len]
        start_positions: 正解の開始位置 [batch_size]
        end_positions: 正解の終了位置 [batch_size]
        
    Returns:
        評価指標の辞書
    """
    start_pred = start_logits.argmax(dim=-1)
    end_pred = end_logits.argmax(dim=-1)
    
    # 開始・終了位置の正確さ
    start_accuracy = (start_pred == start_positions).float().mean().item()
    end_accuracy = (end_pred == end_positions).float().mean().item()
    
    # 両方正解の割合
    exact_match = ((start_pred == start_positions) & (end_pred == end_positions)).float().mean().item()
    
    metrics = {
        "start_accuracy": start_accuracy,
        "end_accuracy": end_accuracy,
        "exact_match": exact_match
    }
    
    return metrics

def compute_tokens_per_second(batch_size: int, seq_length: int, time_seconds: float) -> float:
    """
    処理速度（トークン/秒）を計算
    
    Args:
        batch_size: バッチサイズ
        seq_length: シーケンス長
        time_seconds: 処理時間（秒）
        
    Returns:
        トークン/秒
    """
    return batch_size * seq_length / max(time_seconds, 1e-6)

def compute_flops_efficiency(
    model_flops: float,
    actual_time: float,
    theoretical_flops: float
) -> float:
    """
    FLOPs効率を計算（理論最大性能に対する割合）
    
    Args:
        model_flops: モデルの演算量（FLOPs）
        actual_time: 実際の処理時間
        theoretical_flops: ハードウェアの理論性能（FLOPS/秒）
        
    Returns:
        効率（0.0-1.0）
    """
    actual_flops = model_flops / actual_time
    return min(actual_flops / theoretical_flops, 1.0)  # 最大1.0に制限
