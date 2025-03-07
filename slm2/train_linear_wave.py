#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LinearWaveAttentionを使った線形計算量Wave Networkの学習スクリプト
O(N)の計算量で効率的な波動表現学習を実現
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
from slm.enhanced_wave_models import LinearWaveNetworkLM, HybridWaveNetworkLM
from slm.trainer import Trainer

def main():
    print(f"=== LinearWave Network学習 開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    # シード固定 - 新しいシードを試す
    seed = 7777
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # パス設定
    paths_config = PathsConfig()
    
    # モデル設定 - 少し大きめのモデルを使用
    model_config = ModelConfig(
        hidden_size=384,     # 中規模のサイズ
        num_layers=4,        # より多いレイヤー
        max_seq_len=512,     # 長いシーケンスを処理
        dropout_prob=0.2,    # やや強い正則化
        use_rope=True        # 位置エンコーディング使用
    )
    
    # 学習設定 - より長い学習サイクル
    training_config = TrainingConfig(
        learning_rate=2e-4,     # 少し高めの学習率
        batch_size=32,          # 大きめのバッチ
        mlm_epochs=10,          # より多くのエポック
        mlm_probability=0.15,   # 標準的なマスク率
        weight_decay=0.01,      # 重み減衰あり
        warmup_steps=500,       # 長いウォームアップ
        accumulation_steps=2,   # 勾配累積
        use_amp=True,           # AMP有効で速度向上
    )
    
    # ディレクトリ準備
    linear_wave_dir = os.path.join(paths_config.checkpoint_dir, "linear_wave_model")
    hybrid_wave_dir = os.path.join(paths_config.checkpoint_dir, "hybrid_wave_model")
    os.makedirs(linear_wave_dir, exist_ok=True)
    os.makedirs(hybrid_wave_dir, exist_ok=True)
    
    try:
        if device.type != "cuda":
            print("このスクリプトはGPU環境で実行してください")
            sys.exit(1)
        
        print("=== LinearWaveNetworkLM学習を実行します ===")
        
        # データセット読み込み
        train_dataset = load_from_disk(os.path.join(paths_config.data_dir, "train_dataset"))
        valid_dataset = load_from_disk(os.path.join(paths_config.data_dir, "valid_dataset"))
        
        print(f"完全データセットサイズ: {len(train_dataset)}")
        # 学習用に部分データセット使用
        train_dataset = train_dataset.select(range(min(200000, len(train_dataset))))
        valid_dataset = valid_dataset.select(range(min(10000, len(valid_dataset))))
        print(f"使用データセットサイズ: {len(train_dataset)}")
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(paths_config.tokenizer_name)
        model_config.set_tokenizer(tokenizer)
        
        # シーケンス長を確認・制限する関数
        def check_sequence_lengths(dataset):
            lengths = [len(seq) for seq in dataset['input_ids']]
            max_len = max(lengths)
            avg_len = sum(lengths) / len(lengths)
            print(f"最大シーケンス長: {max_len}, 平均シーケンス長: {avg_len:.1f}")
            
            # 異常に長いシーケンスを検出
            long_seqs = sum(1 for length in lengths if length > 512)
            if long_seqs > 0:
                print(f"警告: 512トークンを超えるシーケンスが{long_seqs}個あります")
        
        # データセットの状態を確認
        print("\n=== データセット統計 ===")
        check_sequence_lengths(train_dataset)
        
        # 1) Linear Wave Networkモデル
        print("\n=== 1. LinearWaveNetworkLM学習 ===")
        linear_model = LinearWaveNetworkLM(model_config)
        
        # 特殊初期化
        for name, param in linear_model.named_parameters():
            if 'attention' in name and 'weight' in name:
                # FFT特性を考慮した初期化
                nn.init.xavier_uniform_(param, gain=0.5)
                
        linear_params = sum(p.numel() for p in linear_model.parameters())
        print(f"LinearWaveNetworkLM パラメータ数: {linear_params:,} ({linear_params/1e6:.2f}M)")
        
        linear_model.to(device)
        
        # LinearWaveNetworkLMのトレーナー
        linear_trainer = Trainer(
            model=linear_model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            training_config=training_config,
            device=device,
            paths_config=paths_config
        )
        
        # 勾配クリッピング(数値安定性のため)
        linear_trainer.clip_value = 1.0
        
        # LinearWaveNetworkLMの学習
        print("LinearWaveNetworkLM学習を開始...")
        linear_trainer.train_mlm()
        
        # 学習済みモデル保存
        linear_trainer.save_checkpoint("linear_wave_model/final_model")
        print(f"LinearWaveNetworkLMモデルを保存しました: {os.path.join(linear_wave_dir, 'final_model.pt')}")
        
        # メモリ解放
        del linear_model
        del linear_trainer
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 2) Hybrid Wave Networkモデル - 両方のアプローチを組み合わせ
        print("\n=== 2. HybridWaveNetworkLM学習 ===")
        hybrid_model = HybridWaveNetworkLM(model_config)
        
        hybrid_params = sum(p.numel() for p in hybrid_model.parameters())
        print(f"HybridWaveNetworkLM パラメータ数: {hybrid_params:,} ({hybrid_params/1e6:.2f}M)")
        
        hybrid_model.to(device)
        
        # Hybridモデルのトレーナー
        hybrid_trainer = Trainer(
            model=hybrid_model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            training_config=training_config,
            device=device,
            paths_config=paths_config
        )
        
        # 勾配クリッピング
        hybrid_trainer.clip_value = 1.0
        
        # HybridWaveNetworkLMの学習
        print("HybridWaveNetworkLM学習を開始...")
        hybrid_trainer.train_mlm()
        
        # 学習済みモデル保存
        hybrid_trainer.save_checkpoint("hybrid_wave_model/final_model")
        print(f"HybridWaveNetworkLMモデルを保存しました: {os.path.join(hybrid_wave_dir, 'final_model.pt')}")
        
        # 結果比較
        print("\n=== モデル性能比較 ===")
        print(f"LinearWaveNetworkLM検証ロス: {linear_trainer.best_val_loss:.4f}")
        print(f"HybridWaveNetworkLM検証ロス: {hybrid_trainer.best_val_loss:.4f}")
        print(f"標準WaveNetworkLM検証ロス: 6.9537 (参考値)")
        print(f"BERT Baseline検証ロス: 6.3034 (参考値)")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # リソース解放
        for var in ['linear_trainer', 'hybrid_trainer']:
            if var in locals():
                locals()[var].close()
        
        print(f"=== 処理完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

if __name__ == "__main__":
    main()
