"""
Wave Networkモデルのトレーニング安定化のためのユーティリティ
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import math

def get_improved_optimizer(model, learning_rate: float = 5e-5, weight_decay: float = 0.01):
    """
    より安定したオプティマイザを返します。
    AdamWにパラメータグループを分け、weight decayを適切に適用
    
    Args:
        model: 対象モデル
        learning_rate: 学習率
        weight_decay: 重み減衰率
        
    Returns:
        最適化されたオプティマイザ
    """
    # バイアス、LayerNorm、Embeddingsのパラメータにはweight decayを適用しない
    no_decay = ["bias", "LayerNorm.weight", "layer_norm", "norm", "embedding"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # AdamWの初期化パラメータ改善
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(0.9, 0.98),  # より安定した値
        eps=1e-6  # 数値安定性向上
    )
    return optimizer

def get_improved_scheduler(optimizer, num_training_steps: int, warmup_steps: int = 500):
    """
    より効果的なスケジューラ: warm up + cosine decay
    
    Args:
        optimizer: オプティマイザ
        num_training_steps: 総トレーニングステップ
        warmup_steps: ウォームアップステップ数
    
    Returns:
        学習率スケジューラ
    """
    def lr_lambda(current_step: int):
        # ウォームアップ部分
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # コサイン減衰部分
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
def initialize_wave_params(module):
    """
    Wave Network に特化したパラメータ初期化
    不安定性解消のための特別な初期化
    
    Args:
        module: 初期化対象のモジュール
    """
    if isinstance(module, nn.Linear):
        # Linearレイヤーは小さな標準偏差で初期化
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
    elif isinstance(module, (nn.LayerNorm, RMSNorm)):
        nn.init.ones_(module.weight)

def gradient_checkpointing_enable(model):
    """
    勾配チェックポイントを有効にしてメモリ使用量を削減
    
    Args:
        model: 対象モデル
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    else:
        # カスタム実装
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        # 各層に勾配チェックポイントを適用
        for i, layer in enumerate(model.layers):
            setattr(model.layers, str(i), torch.utils.checkpoint.checkpoint(create_custom_forward(layer)))

def apply_training_fixes(trainer):
    """
    トレーナーに全ての修正を適用
    
    Args:
        trainer: Trainerインスタンス
    """
    # オプティマイザ改善
    trainer.optimizer = get_improved_optimizer(
        trainer.model, 
        learning_rate=trainer.training_config.learning_rate,
        weight_decay=trainer.training_config.weight_decay
    )
    
    # スケジューラ改善
    total_steps = len(trainer.train_dataset) // trainer.training_config.batch_size * trainer.training_config.mlm_epochs
    trainer.scheduler = get_improved_scheduler(
        trainer.optimizer,
        num_training_steps=total_steps,
        warmup_steps=trainer.training_config.warmup_steps
    )
    
    # 初期化改善
    trainer.model.apply(initialize_wave_params)
    
    # 勾配クリッピング設定
    trainer.clip_value = 1.0
    
    print("トレーニング安定化のための修正を適用しました")
