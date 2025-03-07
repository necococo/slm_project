#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LinearWaveAttentionの効率性と有効性をテストするスクリプト
線形計算量の波動注意機構を検証します
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import load_from_disk
from transformers import AutoTokenizer

from slm.config import ModelConfig, TrainingConfig, PathsConfig
from slm.modules.wave_network import WaveNetworkLM
from slm.slm2.linear_wave_attention import LinearWaveAttention
from slm.trainer import Trainer
from slm.collator import CustomCollator

# LinearWaveAttentionを使った簡易モデル
class LinearWaveModel(nn.Module):
    """LinearWaveAttentionを使った言語モデル"""
    
    def __init__(self, vocab_size, hidden_size=256, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # 位置埋め込み
        self.pos_embedding = nn.Parameter(torch.zeros(1, 512, hidden_size))
        
        # LinearWaveAttentionレイヤー
        self.layers = nn.ModuleList([
            LinearWaveAttention(hidden_size, num_heads=4)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, input_ids):
        # 入力埋め込み
        x = self.embedding(input_ids)
        
        # 位置情報追加 (最大512トークンまで)
        seq_len = input_ids.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # 各レイヤー通過
        for layer in self.layers:
            x = x + layer(x)  # 残差接続
            
        # 最終正規化
        x = self.norm(x)
        
        return x  # hidden_statesを返す
    
    def get_classifier_weights(self):
        """LinearCrossEntropyで必要な分類器の重み"""
        return self.classifier.weight

# 計算量とメモリ使用量を計測する関数
def measure_performance(model, input_size, device, iterations=10):
    """モデルのパフォーマンスを計測"""
    model.eval()
    model = model.to(device)
    
    # ダミー入力
    dummy_input = torch.randint(0, 1000, (1, input_size)).to(device)
    
    # ウォームアップ
    with torch.no_grad():
        _ = model(dummy_input)
    
    torch.cuda.synchronize()
    
    # 推論時間計測
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    inference_time = (time.time() - start_time) / iterations
    
    # メモリ使用量計測
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
        torch.cuda.synchronize()
        memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB単位
    else:
        memory_used = 0
    
    return {
        'inference_time': inference_time,
        'memory_used': memory_used
    }

def main():
    print(f"=== LinearWaveAttention テスト開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # パス設定
    paths_config = PathsConfig()
    os.makedirs(os.path.join(paths_config.log_dir, "linear_wave_test"), exist_ok=True)
    
    try:
        # Tokenizer読み込み
        tokenizer = AutoTokenizer.from_pretrained(paths_config.tokenizer_name)
        vocab_size = len(tokenizer)
        
        # 小さなデータセット読み込み
        try:
            train_dataset = load_from_disk(os.path.join(paths_config.data_dir, "train_dataset"))
            valid_dataset = load_from_disk(os.path.join(paths_config.data_dir, "valid_dataset"))
            
            # テスト用に小さなサブセット
            train_subset = train_dataset.select(range(min(10000, len(train_dataset))))
            valid_subset = valid_dataset.select(range(min(1000, len(valid_dataset))))
        except:
            print("データセットが見つかりません。ダミーデータで実行します。")
            # ダミーデータ作成
            train_subset = {"input_ids": torch.randint(0, vocab_size, (10000, 128))}
            valid_subset = {"input_ids": torch.randint(0, vocab_size, (1000, 128))}
        
        print("\n=== 1. モデル定義 ===")
        # LinearWaveモデル初期化
        linear_wave_model = LinearWaveModel(vocab_size, hidden_size=256, num_layers=2)
        linear_wave_params = sum(p.numel() for p in linear_wave_model.parameters())
        print(f"LinearWaveModel パラメータ数: {linear_wave_params:,}")
        
        # 比較用の標準WaveNetworkLM
        model_config = ModelConfig(hidden_size=256, num_layers=2, vocab_size=vocab_size)
        model_config.set_tokenizer(tokenizer)
        standard_model = WaveNetworkLM(model_config)
        standard_params = sum(p.numel() for p in standard_model.parameters())
        print(f"標準WaveNetworkLM パラメータ数: {standard_params:,}")
        
        print("\n=== 2. 計算量とメモリ使用量の分析 ===")
        sequence_lengths = [128, 256, 512, 1024]
        linear_results = []
        standard_results = []
        
        # 様々なシーケンス長で計測
        for seq_len in sequence_lengths:
            print(f"シーケンス長 {seq_len}:")
            
            # LinearWaveAttention
            linear_perf = measure_performance(linear_wave_model, seq_len, device)
            linear_results.append((seq_len, linear_perf))
            print(f"  LinearWave - 推論時間: {linear_perf['inference_time']*1000:.2f}ms, メモリ使用量: {linear_perf['memory_used']:.2f}MB")
            
            # 標準モデル
            standard_perf = measure_performance(standard_model, seq_len, device)
            standard_results.append((seq_len, standard_perf))
            print(f"  標準モデル - 推論時間: {standard_perf['inference_time']*1000:.2f}ms, メモリ使用量: {standard_perf['memory_used']:.2f}MB")
        
        # 結果をグラフ化
        plt.figure(figsize=(12, 6))
        
        # 推論時間のプロット
        plt.subplot(1, 2, 1)
        plt.plot([r[0] for r in linear_results], [r[1]['inference_time']*1000 for r in linear_results], 'b-o', label='LinearWave')
        plt.plot([r[0] for r in standard_results], [r[1]['inference_time']*1000 for r in standard_results], 'r-o', label='標準モデル')
        plt.xlabel('シーケンス長')
        plt.ylabel('推論時間 (ms)')
        plt.title('シーケンス長と推論時間の関係')
        plt.legend()
        plt.grid(True)
        
        # メモリ使用量のプロット
        plt.subplot(1, 2, 2)
        plt.plot([r[0] for r in linear_results], [r[1]['memory_used'] for r in linear_results], 'b-o', label='LinearWave')
        plt.plot([r[0] for r in standard_results], [r[1]['memory_used'] for r in standard_results], 'r-o', label='標準モデル')
        plt.xlabel('シーケンス長')
        plt.ylabel('メモリ使用量 (MB)')
        plt.title('シーケンス長とメモリ使用量の関係')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(paths_config.log_dir, "linear_wave_test", "performance_comparison.png"))
        print(f"パフォーマンス比較グラフを保存しました: {os.path.join(paths_config.log_dir, 'linear_wave_test', 'performance_comparison.png')}")
        
        print("\n=== 3. 簡易学習テスト ===")
        # LinearWaveモデルの簡易学習テスト
        if isinstance(train_subset, dict):
            print("ダミーデータのため学習テストをスキップします")
        else:
            print("LinearWaveModelの簡易学習を開始...")
            
            # モデル初期化
            linear_wave_model = LinearWaveModel(vocab_size, hidden_size=256, num_layers=2).to(device)
            
            # コレーター定義
            collator = CustomCollator(
                tokenizer=tokenizer,
                model_config=model_config,  # ModelConfigを再利用
                mlm=True,
                mlm_probability=0.15,
                mask_token_id=tokenizer.mask_token_id
            )
            
            # データローダー
            from torch.utils.data import DataLoader
            train_loader = DataLoader(
                train_subset.select(range(1000)),  # より小さなサブセット
                batch_size=16,
                shuffle=True,
                collate_fn=collator
            )
            
            # オプティマイザ
            optimizer = torch.optim.AdamW(linear_wave_model.parameters(), lr=5e-5)
            
            # 簡易学習ループ
            from cut_cross_entropy import linear_cross_entropy
            
            linear_wave_model.train()
            for epoch in range(2):  # 2エポックだけ
                epoch_loss = 0
                for i, batch in enumerate(train_loader):
                    if i >= 20:  # 最初の20バッチだけ
                        break
                    
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    
                    optimizer.zero_grad()
                    
                    # フォワードパス
                    embeddings = linear_wave_model(input_ids)
                    classifier = linear_wave_model.get_classifier_weights()
                    
                    # 半精度に変換
                    embeddings = embeddings.half()
                    classifier = classifier.half()
                    
                    # ロス計算
                    loss = linear_cross_entropy(embeddings, classifier, labels)
                    loss.backward()
                    
                    # 勾配クリッピング
                    torch.nn.utils.clip_grad_norm_(linear_wave_model.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    if i % 5 == 0:
                        print(f"Epoch {epoch+1}/2 | Batch {i}/20 | Loss: {loss.item():.6f}")
                
                avg_loss = epoch_loss / min(20, len(train_loader))
                print(f"Epoch {epoch+1}/2 | Avg Loss: {avg_loss:.6f}")
            
            print("LinearWaveModelの簡易学習が完了しました")
        
        print("\n=== 4. 長いシーケンスでの挙動テスト ===")
        # 非常に長いシーケンスでもエラーなく処理できるか
        try:
            very_long_seq = 2048
            dummy_input = torch.randint(0, vocab_size, (1, very_long_seq)).to(device)
            
            with torch.no_grad():
                # LinearWaveModel
                start_time = time.time()
                _ = linear_wave_model(dummy_input)
                linear_time = time.time() - start_time
                print(f"LinearWaveModel - 長さ{very_long_seq}の処理時間: {linear_time:.4f}秒")
                
                try:
                    # 標準モデル（メモリ不足でエラーになる可能性あり）
                    start_time = time.time()
                    _ = standard_model(dummy_input)
                    standard_time = time.time() - start_time
                    print(f"標準WaveNetworkLM - 長さ{very_long_seq}の処理時間: {standard_time:.4f}秒")
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"標準WaveNetworkLMはメモリ不足エラーが発生: {str(e)}")
                    else:
                        print(f"標準WaveNetworkLMでエラーが発生: {str(e)}")
        except Exception as e:
            print(f"長いシーケンステストでエラーが発生: {str(e)}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"=== テスト完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

if __name__ == "__main__":
    main()
