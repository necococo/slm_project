#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
小型Wave Networkモデルのトレーニング
SwiGLUの特性を活かした効率的な設計

このスクリプトはmain.pyと同様の流れで、より小さなモデルをトレーニングします
"""

import os
import sys
import torch
from datetime import datetime
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

from slm.config import PathsConfig
from slm.configs.tiny_model import get_tiny_config, get_tiny_training_config
from slm.modules.wave_network import WaveNetworkLM
from slm.trainer import Trainer

def main():
    print(f"=== 小型Wave Network言語モデル学習 開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # パス設定
    paths_config = PathsConfig()
    
    # 小型モデル設定
    model_config = get_tiny_config()
    
    # 小型モデルのトレーニング設定
    training_config = get_tiny_training_config()
    
    # ディレクトリ準備
    tiny_model_dir = os.path.join(paths_config.checkpoint_dir, "tiny_model")
    os.makedirs(tiny_model_dir, exist_ok=True)
    os.makedirs(paths_config.log_dir, exist_ok=True)
    
    try:
        # GPUモードでのトレーニングを想定（前処理済みデータを使用）
        if device.type == "cuda":
            print("=== GPU環境: 学習を実行します ===")
            
            # 前処理済みデータをロード
            train_dataset = load_from_disk(os.path.join(paths_config.data_dir, "train_dataset"))
            valid_dataset = load_from_disk(os.path.join(paths_config.data_dir, "valid_dataset"))
            
            # Tokenizer読み込み
            tokenizer = AutoTokenizer.from_pretrained(paths_config.tokenizer_name)
            
            # model_configにトークナイザをセット
            model_config.set_tokenizer(tokenizer)
            
            # 小型WaveNetworkLM を初期化
            print("小型WaveNetworkLMを初期化します...")
            model = WaveNetworkLM(model_config)
            
            # パラメータ数の報告
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"モデルパラメータ数: {total_params:,} ({total_params/1e6:.2f}M)")

            # トレーナー
            trainer = Trainer(
                model=model,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                training_config=training_config,
                device=device,
                paths_config=paths_config
            )
            
            # MLM学習
            print("MLM学習を開始...")
            trainer.train_mlm()
            
            # チェックポイント保存
            final_model_path = os.path.join(tiny_model_dir, "final_model.pt")
            trainer.save_checkpoint("tiny_model/final_model")
            print(f"モデルを保存しました: {final_model_path}")
        else:
            print("GPUが必要です。CPUモードではデータ前処理のみ実行可能です。")
            print("先に main.py でデータ前処理を行ってください。")
            sys.exit(1)
        
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
