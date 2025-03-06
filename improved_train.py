#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改良版Wave Network トレーニングスクリプト
安定したロス減少を実現するための各種改善を適用
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
from datasets import load_from_disk
from transformers import AutoTokenizer

from slm.config import ModelConfig, TrainingConfig, PathsConfig
from slm.modules.wave_network import WaveNetworkLM
from slm.trainer import Trainer
from slm.training_fixes import apply_training_fixes

def main():
    print(f"=== 改良版Wave Network学習 開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    # 再現性のために乱数シードを固定
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
    
    # 改善されたモデル設定
    model_config = ModelConfig(
        hidden_size=768,
        num_layers=3,
        max_seq_len=512,
        # ロスが下がらない問題に対処するため、ドロップアウト率を調整
        dropout_prob=0.1,
        use_rope=True
    )
    
    # 改善されたトレーニング設定
    training_config = TrainingConfig(
        # 学習率を少し上げる
        learning_rate=5e-5,
        batch_size=64,
        mlm_epochs=5,
        # MLMマスク確率を下げる
        mlm_probability=0.15,
        weight_decay=0.01,
        warmup_steps=1000,
        # 勾配累積でバッチを大きく見せる
        accumulation_steps=4,
        use_amp=True,
    )
    
    # ディレクトリ準備
    improved_model_dir = os.path.join(paths_config.checkpoint_dir, "improved_model")
    os.makedirs(improved_model_dir, exist_ok=True)
    
    try:
        if device.type != "cuda":
            print("このスクリプトはGPU環境で実行してください。")
            sys.exit(1)
        
        print("=== 改良版学習を実行します ===")
        
        # 前処理済みデータをロード
        train_dataset = load_from_disk(os.path.join(paths_config.data_dir, "train_dataset"))
        valid_dataset = load_from_disk(os.path.join(paths_config.data_dir, "valid_dataset"))
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(paths_config.tokenizer_name)
        model_config.set_tokenizer(tokenizer)
        
        # モデル
        model = WaveNetworkLM(model_config)
        
        # トレーナー
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            training_config=training_config,
            device=device,
            paths_config=paths_config
        )
        
        # ※重要※ 安定化のための修正を適用
        apply_training_fixes(trainer)
        
        # MLM学習
        print("改良版学習を開始...")
        trainer.train_mlm()
        
        # 保存
        final_model_path = os.path.join(improved_model_dir, "final_model.pt")
        trainer.save_checkpoint("improved_model/final_model")
        print(f"改良版モデルを保存しました: {final_model_path}")
        
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
