#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
強化版Wave Networkトレーニングスクリプト
トークン間の関係を波動表現で直接モデル化
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from datasets import load_from_disk
from transformers import AutoTokenizer

from slm.config import ModelConfig, TrainingConfig, PathsConfig
from slm.modules.wave_network import WaveNetworkLM
from slm.train import Trainer
from slm.collator import CustomCollator

def main():
    print(f"=== 強化版Wave Network学習 開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    # シード固定
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 設定
    paths_config = PathsConfig()
    
    # トークン間関係を波動で直接モデル化するためのモデル設定
    model_config = ModelConfig(
        hidden_size=384,    # 十分な表現力のサイズ
        num_layers=3,       # 複雑な関係を捉えるのに十分な深さ
        max_seq_len=256,    # 中程度の文脈長
        dropout_prob=0.1,   # 適度な正則化
        use_rope=True       # 位置情報は重要
    )
    
    # 学習設定
    training_config = TrainingConfig(
        learning_rate=5e-5,
        batch_size=16,
        mlm_epochs=5,
        mlm_probability=0.15,
        weight_decay=0.01,
        warmup_steps=500
    )
    
    # ディレクトリ準備
    enhanced_dir = os.path.join(paths_config.checkpoint_dir, "enhanced_wave_model")
    os.makedirs(enhanced_dir, exist_ok=True)
    
    try:
        if device.type != "cuda":
            print("このスクリプトはGPU環境で実行してください")
            sys.exit(1)
        
        print("=== 強化版Wave Network学習を実行します ===")
        
        # データセット読み込み
        train_dataset = load_from_disk(os.path.join(paths_config.data_dir, "train_dataset"))
        valid_dataset = load_from_disk(os.path.join(paths_config.data_dir, "valid_dataset"))
        
        print(f"元のデータセットサイズ: {len(train_dataset)}")
        # 学習用サブセット
        train_subset = train_dataset.select(range(min(5000, len(train_dataset))))
        valid_subset = valid_dataset.select(range(min(50, len(valid_dataset))))
        print(f"使用データセットサイズ: {len(train_subset)}")
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(paths_config.tokenizer_name)
        model_config.set_tokenizer(tokenizer)
        
        # 強化版モデルの初期化
        model = WaveNetworkLM(model_config)
        print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()) / 1000000:.2f}M")
        
        # 初期化の調整
        for name, param in model.named_parameters():
            if 'wave_interaction' in name or 'wavelet_attention' in name or 'phase_interaction' in name:
                if 'weight' in name:
                    # 波動関係モデリング層は特別な初期化
                    torch.nn.init.xavier_uniform_(param, gain=0.2)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
        
        model.to(device)
        
        # トレーナー
        trainer = Trainer(
            model=model,
            train_dataset=train_subset,
            valid_dataset=valid_subset,
            training_config=training_config,
            device=device,
            paths_config=paths_config
        )
        
        # 勾配クリッピング設定
        trainer.clip_value = 1.0
        
        # 学習開始
        print("強化版Wave Network学習を開始...")
        trainer.train_mlm()
        
        # モデル保存
        trainer.save_checkpoint("enhanced_wave_model/checkpoint")
        print(f"強化版モデルを保存しました: {os.path.join(enhanced_dir, 'checkpoint.pt')}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'trainer' in locals():
            trainer.close()
        
        print(f"=== 処理完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

if __name__ == "__main__":
    main()
