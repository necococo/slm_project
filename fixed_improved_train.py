#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
より強化されたWave Network学習スクリプト
ロスが下がらない問題に特化した修正を追加
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
    print(f"=== 強化版Wave Network学習 開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    # シード固定
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 設定
    paths_config = PathsConfig()
    
    # 小さくシンプルな設定から始める
    model_config = ModelConfig(
        hidden_size=384,  # サイズを小さくする
        num_layers=2,     # レイヤー数を減らす
        max_seq_len=256,  # シーケンス長を短くする
        dropout_prob=0.1, 
        use_rope=True
    )
    
    # より積極的な学習設定
    training_config = TrainingConfig(
        learning_rate=1e-4,  # より大きな学習率
        batch_size=32,       # バッチサイズを小さく
        mlm_epochs=1,        # まずは1エポックでテスト
        mlm_probability=0.15,
        weight_decay=0.01,
        warmup_steps=100,    # ウォームアップを短く
        accumulation_steps=2,
        use_amp=True,
    )
    
    # ディレクトリ準備
    fixed_model_dir = os.path.join(paths_config.checkpoint_dir, "fixed_model")
    os.makedirs(fixed_model_dir, exist_ok=True)
    
    try:
        if device.type != "cuda":
            print("このスクリプトはGPU環境で実行してください。")
            sys.exit(1)
        
        print("=== 強化版学習を実行します ===")
        
        # サンプリングを減らして処理を高速化（テスト用）
        train_dataset = load_from_disk(os.path.join(paths_config.data_dir, "train_dataset"))
        valid_dataset = load_from_disk(os.path.join(paths_config.data_dir, "valid_dataset"))
        
        # データセットを小さくしてテスト（開発中）
        print(f"元のデータセットサイズ: {len(train_dataset)}")
        train_dataset = train_dataset.select(range(min(100000, len(train_dataset))))
        valid_dataset = valid_dataset.select(range(min(1000, len(valid_dataset))))
        print(f"テスト用に縮小: {len(train_dataset)}")
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(paths_config.tokenizer_name)
        model_config.set_tokenizer(tokenizer)
        
        # モデル初期化（重み初期化を特別に調整）
        model = WaveNetworkLM(model_config)
        
        # 初期化を手動で調整
        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'norm' in name or 'layer_norm' in name:
                    # ノーマライゼーション層は1で初期化
                    torch.nn.init.ones_(param)
                elif 'embedding' in name:
                    # 埋め込み層は小さな値で初期化
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
                else:
                    # その他の重みは小さな値で初期化
                    torch.nn.init.xavier_uniform_(param, gain=0.01)
            elif 'bias' in name:
                # バイアスは0で初期化
                torch.nn.init.zeros_(param)
            elif 'scale' in name:
                # RMSNormのスケールは1で初期化
                torch.nn.init.ones_(param)
        
        # トレーナー（learning_rate_override でトレーナー内の警告を無効化）
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            training_config=training_config,
            device=device,
            paths_config=paths_config
        )
        
        # 勾配クリッピング値を設定（training_fixesの値が小さすぎる場合がある）
        trainer.clip_value = 5.0
        
        # 安定化修正を適用
        apply_training_fixes(trainer)
        
        # 学習率を上書き設定（training_fixesが下げる場合がある）
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = 1e-4
            print(f"学習率を設定: {param_group['lr']}")
        
        # 学習開始（一旦短いエポック数で試す）
        print("強化版学習を開始...")
        trainer.train_mlm()
        
        # チェックポイント保存
        trainer.save_checkpoint("fixed_model/checkpoint")
        print(f"強化版モデルを保存しました: {os.path.join(fixed_model_dir, 'checkpoint.pt')}")
        
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
