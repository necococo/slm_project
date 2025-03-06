#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改善版Wave Network言語モデル学習スクリプト
ロスが7付近で停滞する問題に対処
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

def main():
    print(f"=== 改善版Wave Network学習 開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    # シード固定
    seed = 123  # 別のシードを試す
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 設定
    paths_config = PathsConfig()
    
    # より深いモデル設定
    model_config = ModelConfig(
        # より大きなモデルサイズ
        hidden_size=512,  
        # より深いネットワーク
        num_layers=4,     
        # シーケンス長を増やす
        max_seq_len=512,  
        # ドロップアウトを増やして過学習を防ぐ
        dropout_prob=0.3, 
        # 位置符号化は必須
        use_rope=True
    )
    
    # 改善されたトレーニング設定
    training_config = TrainingConfig(
        # より小さな学習率
        learning_rate=5e-6,
        # 小さいバッチサイズ
        batch_size=24,
        # より多くのエポック
        mlm_epochs=5,
        # マスク確率
        mlm_probability=0.15,
        weight_decay=0.02,
        # より長いウォームアップ
        warmup_steps=2000,
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
        
        print("=== 改善版学習を実行します ===")
        
        # データセットをロード（一部だけ使用）
        train_dataset = load_from_disk(os.path.join(paths_config.data_dir, "train_dataset"))
        valid_dataset = load_from_disk(os.path.join(paths_config.data_dir, "valid_dataset"))
        
        print(f"元のデータセットサイズ: {len(train_dataset)}")
        # 学習を早くするため、データセットサイズを小さくする（本番環境では全データを使用）
        train_dataset = train_dataset.select(range(min(200000, len(train_dataset))))
        valid_dataset = valid_dataset.select(range(min(2000, len(valid_dataset))))
        print(f"使用データセットサイズ: {len(train_dataset)}")
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(paths_config.tokenizer_name)
        model_config.set_tokenizer(tokenizer)
        
        # モデル初期化
        model = WaveNetworkLM(model_config)
        
        # 特別な初期化
        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'norm' in name or 'layer_norm' in name or 'scale' in name:
                    # ノーマライゼーション層は1で初期化
                    torch.nn.init.ones_(param)
                elif 'embedding' in name:
                    # 埋め込み層は小さな値で初期化
                    torch.nn.init.normal_(param, mean=0.0, std=0.01)
                else:
                    # 通常のレイヤー（小さな値で初期化）
                    torch.nn.init.xavier_normal_(param, gain=0.7)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        
        # トレーナー
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            training_config=training_config,
            device=device,
            paths_config=paths_config
        )
        
        # 強い勾配クリッピング
        trainer.clip_value = 1.0
        
        # 学習開始
        print("改善版学習を開始...")
        trainer.train_mlm()
        
        # チェックポイント保存
        trainer.save_checkpoint("improved_retry_model")
        print(f"改善版モデルを保存しました")
        
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
