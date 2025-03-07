#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wave Network抜本的修正版トレーニングスクリプト
ロスが7付近で停滞する問題に根本対策を施します
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from datasets import load_from_disk
from transformers import AutoTokenizer  # transformers直接使用に変更
from torch.utils.data import DataLoader

from slm.config import ModelConfig, TrainingConfig, PathsConfig
from slm.modules.wave_network import WaveNetworkLM
from slm.trainer import Trainer
from slm.debug_wave import WaveDebugger
from slm.collator import CustomCollator
from slm.inference import sample_text

def main():
    print(f"=== Wave Network抜本修正版学習 開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    # シード設定
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # パス設定
    paths_config = PathsConfig(
    )
    
    # モデル設定
    model_configs = [ModelConfig(
    )]
    
    # 学習設定
    training_configs = [TrainingConfig(
    )]
    
    # ディレクトリ作成
    os.makedirs(paths_config.data_dir, exist_ok=True)
    os.makedirs(paths_config.checkpoint_dir, exist_ok=True)
    os.makedirs(paths_config.log_dir, exist_ok=True)
    deep_fix_dir = os.path.join(paths_config.checkpoint_dir, "deep_fix_model")
    os.makedirs(deep_fix_dir, exist_ok=True)
    
    # もし'resume'引数が指定されていたら、保存されたチェックポイントから復元する
    resume_checkpoint = None
    if len(sys.argv) > 1 and sys.argv[1] == "resume":
        resume_checkpoint = os.path.join(paths_config.checkpoint_dir, "final_model.pt")
        print(f"チェックポイントから復元します: {resume_checkpoint}")
    
    try:
        # データセット読み込み
        print("データセットをロード中...")
        try:
            train_dataset = load_from_disk(os.path.join(paths_config.data_dir, "train_dataset"))
            valid_dataset = load_from_disk(os.path.join(paths_config.data_dir, "valid_dataset"))
        except Exception as e:
            print(f"データセットの読み込みエラー: {e}")
            print("既存のデータセットが見つかりません。CPU環境での前処理が必要です。")
            sys.exit(1)
        
        # トークナイザー読み込み（transformers AutoTokenizerを直接使用）
        print("トークナイザーをロード中...")
        try:
            # AutoTokenizerを使用
            tokenizer = AutoTokenizer.from_pretrained(paths_config.tokenizer_name)
            print(f"Hugging Face tokenizer loaded: {paths_config.tokenizer_name}")
            print(f"語彙サイズ: {len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else tokenizer.vocab_size}")
        except Exception as e:
            print(f"トークナイザー読み込みエラー: {e}")
            sys.exit(1)
            
        print(f"データセットサイズ: {len(train_dataset)}")
        
        # テスト用のバッチを取得（勾配フロー解析用）
        collator = CustomCollator(
            tokenizer=tokenizer, 
            model_config=model_configs[0],
            mlm=True,
            mlm_probability=0.15,
            mask_token_id=tokenizer.mask_token_id
        )
        
        # 実験ループ
        for model_idx, model_config in enumerate(model_configs):
            print(f"\n=== モデル構成 {model_idx+1}/{len(model_configs)} ===")
            print(f"hidden_size: {model_config.hidden_size}, layers: {model_config.num_layers}, norm: {model_config.norm_scheme}")
            
            # トークナイザー設定（直接AutoTokenizerオブジェクトを設定）
            model_config.set_tokenizer(tokenizer)
            
            for train_idx, training_config in enumerate(training_configs):
                print(f"\n--- 学習構成 {train_idx+1}/{len(training_configs)} ---")
                
                # モデル初期化
                model = WaveNetworkLM(model_config)
                
                # もしチェックポイントが存在すれば復元
                if resume_checkpoint and os.path.exists(resume_checkpoint):
                    from slm.utils import load_checkpoint
                    load_checkpoint(resume_checkpoint, model)  # optimizer等も必要なら適宜復元してください
                
                # トレーナー初期化
                trainer = Trainer(
                    model=model,
                    train_dataset=train_dataset,
                    valid_dataset=valid_dataset,
                    training_config=training_config,
                    device=device,
                    paths_config=paths_config
                )
                
                # MLM学習
                print("MLM学習を開始します...")
                trainer.train_mlm()
                
                # 最終チェックポイント保存
                final_model_path = os.path.join(paths_config.checkpoint_dir, "final_model.pt")
                trainer.save_checkpoint("final_model")
                print(f"モデルを保存しました: {final_model_path}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # リソース解放
        if 'trainer' in locals():
            trainer.close()
        
        print(f"=== 処理完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")


if __name__ == "__main__":
    main()