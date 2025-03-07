#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wave Network用カリキュラム学習スクリプト
簡単なタスクから徐々に難しくしていく学習方法を実装
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from datasets import load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset

from slm.config import ModelConfig, TrainingConfig, PathsConfig
from slm.modules.wave_network import WaveNetworkLM
from slm.train import Trainer
from slm.collator import CustomCollator
from slm.debug_wave import WaveDebugger

def main():
    print(f"=== Wave Network カリキュラム学習 開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # シード固定
    seed = 2023
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 設定
    paths_config = PathsConfig()
    
    # ディレクトリ準備
    curriculum_dir = os.path.join(paths_config.checkpoint_dir, "curriculum_model")
    os.makedirs(curriculum_dir, exist_ok=True)
    
    try:
        if device.type != "cuda":
            print("このスクリプトはGPU環境で実行してください")
            sys.exit(1)
            
        # データセット読み込み
        train_dataset = load_from_disk(os.path.join(paths_config.data_dir, "train_dataset"))
        valid_dataset = load_from_disk(os.path.join(paths_config.data_dir, "valid_dataset"))
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(paths_config.tokenizer_name)
        
        print("カリキュラム学習を開始します")
        
        # ステージ1: 超小型モデル + 超小さいデータセット + シンプルな目標
        print("\n=== ステージ1: 基本的な埋め込み学習 ===")
        model_config_1 = ModelConfig(
            hidden_size=128,
            num_layers=1,
            max_seq_len=64,
            dropout_prob=0.0,
            use_rope=False  # まずは位置エンコーディングなし
        )
        model_config_1.set_tokenizer(tokenizer)
        
        model_1 = WaveNetworkLM(model_config_1)
        
        # 重要: カスタム初期化（非常に小さい値）
        for name, param in model_1.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0.0, std=0.01)
            elif "bias" in name:
                nn.init.zeros_(param)
        
        model_1.to(device)
        
        # 極小データセット (1000サンプルだけ)
        tiny_dataset = train_dataset.select(range(1000))
        tiny_valid = valid_dataset.select(range(100))
        
        # シンプルなロス関数とトレーニング
        optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=0.001)
        
        collator = CustomCollator(
            tokenizer=tokenizer,
            model_config=model_config_1,
            mlm=True,
            mlm_probability=0.15,
            mask_token_id=tokenizer.mask_token_id
        )
        
        dataloader = DataLoader(
            tiny_dataset, 
            batch_size=8,
            shuffle=True,
            collate_fn=collator
        )
        
        # シンプルなMSEトレーニング
        print("ステージ1学習中...")
        model_1.train()
        for epoch in range(3):
            epoch_loss = 0
            for i, batch in enumerate(dataloader):
                if i >= 100:  # 最初の100バッチだけ
                    break
                    
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                optimizer_1.zero_grad()
                
                # シンプルなMSEトレーニングでembeddingのみを学習
                out = model_1(input_ids)
                
                # ラベルのembedding取得
                target = torch.zeros_like(out)
                mask = (labels != -100)
                
                for b in range(labels.size(0)):
                    for s in range(labels.size(1)):
                        if labels[b, s] >= 0:
                            target[b, s] = model_1.token_embedding(labels[b, s]).detach()
                
                # MSE損失
                loss = nn.MSELoss()(out * mask.unsqueeze(-1).float(), target * mask.unsqueeze(-1).float())
                loss.backward()
                
                # 勾配クリッピング
                nn.utils.clip_grad_norm_(model_1.parameters(), 1.0)
                optimizer_1.step()
                
                epoch_loss += loss.item()
                
                if i % 10 == 0:
                    print(f"Stage 1 - Epoch {epoch+1}/3 | Batch {i}/100 | Loss: {loss.item():.6f}")
            
            print(f"Stage 1 - Epoch {epoch+1}/3 | Avg Loss: {epoch_loss/(i+1):.6f}")
        
        # ステージ1のモデルを保存
        torch.save(model_1.state_dict(), os.path.join(curriculum_dir, "stage1.pt"))
        
        # ステージ2: 小型モデル + RoPEを追加
        print("\n=== ステージ2: 位置エンコーディング追加 ===")
        model_config_2 = ModelConfig(
            hidden_size=256,
            num_layers=1,
            max_seq_len=128,
            dropout_prob=0.1,
            use_rope=True  # 位置エンコーディング追加
        )
        model_config_2.set_tokenizer(tokenizer)
        
        model_2 = WaveNetworkLM(model_config_2)
        
        # ステージ1から共通パラメータをコピー
        # (埋め込み層のみ - サイズが違うので注意)
        embedding_1 = model_1.token_embedding.weight.data
        embedding_2 = model_2.token_embedding.weight.data
        
        # サイズが異なる場合は共通部分だけコピー
        min_dim = min(embedding_1.size(1), embedding_2.size(1))
        model_2.token_embedding.weight.data[:, :min_dim] = embedding_1[:, :min_dim]
        
        model_2.to(device)
        
        # トレーニング設定
        training_config_2 = TrainingConfig(
            learning_rate=5e-5,
            batch_size=16,
            mlm_epochs=3,
            mlm_probability=0.15,
            weight_decay=0.01,
            warmup_steps=100
        )
        
        # 小さいデータセット
        small_dataset = train_dataset.select(range(10000))
        small_valid = valid_dataset.select(range(1000))
        
        # トレーナー
        trainer_2 = Trainer(
            model=model_2,
            train_dataset=small_dataset,
            valid_dataset=small_valid,
            training_config=training_config_2,
            device=device,
            paths_config=paths_config
        )
        
        # 勾配クリッピング設定
        trainer_2.clip_value = 1.0
        
        # MLM学習
        print("ステージ2学習中...")
        trainer_2.train_mlm()
        
        # ステージ2のモデルを保存
        trainer_2.save_checkpoint("curriculum_model/stage2")
        
        # ステージ3: 標準モデル + 大きめのデータセット
        print("\n=== ステージ3: フルモデルトレーニング ===")
        model_config_3 = ModelConfig(
            hidden_size=384,
            num_layers=3,
            max_seq_len=256,
            dropout_prob=0.1,
            use_rope=True
        )
        model_config_3.set_tokenizer(tokenizer)
        
        model_3 = WaveNetworkLM(model_config_3)
        model_3.to(device)
        
        # トレーニング設定
        training_config_3 = TrainingConfig(
            learning_rate=1e-5,
            batch_size=32,
            mlm_epochs=5,
            mlm_probability=0.15,
            weight_decay=0.01,
            warmup_steps=500
        )
        
        # より大きいデータセット
        medium_dataset = train_dataset.select(range(100000))
        medium_valid = valid_dataset.select(range(5000))
        
        # トレーナー
        trainer_3 = Trainer(
            model=model_3,
            train_dataset=medium_dataset,
            valid_dataset=medium_valid,
            training_config=training_config_3,
            device=device,
            paths_config=paths_config
        )
        
        # 勾配クリッピング設定
        trainer_3.clip_value = 1.0
        
        # MLM学習
        print("ステージ3学習中...")
        trainer_3.train_mlm()
        
        # 最終モデルを保存
        trainer_3.save_checkpoint("curriculum_model/final_model")
        print(f"カリキュラム学習完了。最終モデルを保存: {os.path.join(curriculum_dir, 'final_model.pt')}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for var in ['trainer_2', 'trainer_3']:
            if var in locals():
                locals()[var].close()
    
    print(f"=== 処理完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

if __name__ == "__main__":
    main()
