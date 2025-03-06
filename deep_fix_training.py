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
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from slm.config import ModelConfig, TrainingConfig, PathsConfig
from slm.modules.wave_network import WaveNetworkLM
from slm.trainer import Trainer
from slm.debug_wave import WaveDebugger
from slm.collator import CustomCollator

def main():
    print(f"=== Wave Network抜本修正版学習 開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    # 再現性のために強固なシード固定
    seed = 3407  # "Seed 3407 is all you need" 論文からの数値
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 設定
    paths_config = PathsConfig()
    
    # モデル設定: 極小モデルからスタート
    model_config = ModelConfig(
        hidden_size=256,     # 小さいモデルサイズ
        num_layers=1,        # 少ないレイヤー数
        max_seq_len=128,     # 短いシーケンス
        dropout_prob=0.0,    # まずドロップアウトをなしで試す
        use_rope=True        # 位置エンコーディングは必須
    )
    
    # トレーニング設定: 大胆な設定
    training_config = TrainingConfig(
        learning_rate=1e-3,     # 非常に大きい学習率でスタート
        batch_size=16,          # 小さなバッチサイズ
        mlm_epochs=10,          # より多くのエポック
        mlm_probability=0.15,   # 標準的なマスク率
        weight_decay=0.0,       # ウェイト減衰なし
        warmup_steps=100,       # 短いウォームアップ
        accumulation_steps=1,   # 勾配累積なし
        use_amp=False,          # AMP無効（数値安定性のため）
    )
    
    # ディレクトリ準備
    deep_fix_dir = os.path.join(paths_config.checkpoint_dir, "deep_fix_model")
    os.makedirs(deep_fix_dir, exist_ok=True)
    os.makedirs(os.path.join(paths_config.log_dir, "wave_debug"), exist_ok=True)
    
    try:
        if device.type != "cuda":
            print("このスクリプトはGPU環境で実行してください。")
            sys.exit(1)
        
        print("=== 抜本修正版学習を実行します ===")
        
        # 非常に小さなデータセットから始める
        train_dataset = load_from_disk(os.path.join(paths_config.data_dir, "train_dataset"))
        valid_dataset = load_from_disk(os.path.join(paths_config.data_dir, "valid_dataset"))
        
        print(f"元のデータセットサイズ: {len(train_dataset)}")
        # 極小データセットで最初に実験
        train_dataset_small = train_dataset.select(range(min(10000, len(train_dataset))))
        valid_dataset_small = valid_dataset.select(range(min(1000, len(valid_dataset))))
        print(f"実験用データセットサイズ: {len(train_dataset_small)}")
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(paths_config.tokenizer_name)
        model_config.set_tokenizer(tokenizer)
        
        # モデル初期化: 特殊な初期化
        model = WaveNetworkLM(model_config)
        
        # 誰もが見落とす重要なポイント: 非常に慎重な初期化
        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'embedding' in name:
                    # 埋め込み層は通常初期化
                    nn.init.normal_(param, mean=0.0, std=0.02)
                elif 'norm' in name or 'scale' in name:
                    # ノーマライゼーション層
                    nn.init.constant_(param, 1.0)
                elif 'classifier' in name:
                    # 出力層は非常に小さく初期化
                    nn.init.normal_(param, mean=0.0, std=0.001)
                else:
                    # 重み行列全般は非常に慎重に初期化
                    nn.init.xavier_uniform_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
            
        # Step 1: 特殊な直接トレーニング - カスタム実装で最初の数バッチを特別扱い
        print("Step 1: 特殊な初期ウォームアップトレーニング...")
        
        # カスタムコレーター
        collator = CustomCollator(
            tokenizer=tokenizer,
            model_config=model_config,
            mlm=True,
            mlm_probability=training_config.mlm_probability,
            mask_token_id=tokenizer.mask_token_id,
            qa=False
        )
        
        dataloader = DataLoader(
            train_dataset_small,
            batch_size=training_config.batch_size,
            shuffle=True,
            collate_fn=collator
        )
        
        # カスタム最適化器と学習率
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        model.to(device)
        
        # 最初のウォームアップバッチ - 非常に単純な損失関数
        for epoch in range(2):  # 2エポックだけ
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 50:  # 最初の50バッチだけ
                    break
                
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                # 単純な損失関数 - 埋め込みとコサイン類似度
                optimizer.zero_grad()
                out = model(input_ids)
                
                # 単純なMSE損失
                target = torch.zeros_like(out)
                for i in range(len(labels)):
                    for j in range(len(labels[i])):
                        if labels[i, j] >= 0:  # -100でない場合のみ
                            label_vec = model.token_embedding(labels[i, j]).detach()
                            target[i, j, :] = label_vec
                
                # 損失計算 (単純なMSEで埋め込みスペースを学習)
                loss = nn.MSELoss()(out, target)
                loss.backward()
                
                # 大きな勾配をクリップ
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                if batch_idx % 10 == 0:
                    print(f"Warmup Epoch {epoch+1}/2 | Batch {batch_idx}/50 | Loss: {loss.item():.6f}")
        
        # Step 2: 診断
        print("\nStep 2: Wave表現の診断...")
        
        # 波表現を診断
        sample_batch = next(iter(dataloader))
        sample_input = sample_batch["input_ids"].to(device)
        
        wave_stats = WaveDebugger.check_wave_representation(
            model, 
            sample_input, 
            save_path=os.path.join(paths_config.log_dir, "wave_debug", "wave_repr_hist.png")
        )
        
        # 問題のあるレイヤーを検出
        problematic_layers = []
        for name, stats in wave_stats.items():
            if stats['has_nan_real'] or stats['has_nan_imag'] or stats['has_inf_real'] or stats['has_inf_imag']:
                problematic_layers.append(name)
                print(f"警告: {name}に数値的問題があります。NaN or Inf検出。")
        
        # Step 3: 標準トレーナーでの学習
        print("\nStep 3: 主要トレーニングフェーズ...")
        
        # トレーナー
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset_small,  # 小さなデータセットで始める
            valid_dataset=valid_dataset_small,
            training_config=training_config,
            device=device,
            paths_config=paths_config
        )
        
        # 強い勾配クリッピング
        trainer.clip_value = 1.0
        
        # 学習開始
        print("抜本修正版学習を開始...")
        trainer.train_mlm(num_epochs=3)  # まず3エポックだけ
        
        # Step 4: 勾配解析とトレーニング調整
        print("\nStep 4: 勾配解析とトレーニング調整...")
        
        # 勾配解析
        grad_stats = WaveDebugger.analyze_gradients(
            model, trainer.optimizer, dataloader, trainer, num_steps=5, clip_value=1.0
        )
        
        # 結果報告
        for i, stats in enumerate(grad_stats):
            print(f"バッチ {i} | ロス: {stats['loss']:.4f} | 勾配ノルム: {stats['overall_grad_norm']:.4f}")
            
            # 異常な勾配を検出
            problem_layers = []
            for name, grad_info in stats['layer_gradients'].items():
                if grad_info['has_nan'] or grad_info['has_inf'] or grad_info['norm'] > 10.0:
                    problem_layers.append(name)
            
            if problem_layers:
                print(f"  問題のあるレイヤー: {', '.join(problem_layers)}")
        
        # Step 5: フルデータでの学習再開
        print("\nStep 5: フルデータでの学習再開...")
        
        # 学習率の再調整
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = 5e-5
        
        # トレーナーの設定更新
        trainer.train_dataset = train_dataset.select(range(min(100000, len(train_dataset))))
        trainer.valid_dataset = valid_dataset.select(range(min(5000, len(valid_dataset))))
        print(f"拡大データセットサイズ: {len(trainer.train_dataset)}")
        
        # 残りのエポックを学習
        trainer.train_mlm(num_epochs=7)
        
        # チェックポイント保存
        trainer.save_checkpoint("deep_fix_model/final_model")
        print(f"修正版モデルを保存しました: {os.path.join(deep_fix_dir, 'final_model.pt')}")
        
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
