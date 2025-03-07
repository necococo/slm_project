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
from slm.inference import sample_text

# 追加: 超小規模データセット作成関数
def create_micro_dataset(dataset, size=100, vocab_size=1000):
    """超小規模データセットを作成して問題の再現性を確認"""
    micro_data = dataset.select(range(min(size, len(dataset))))
    
    # さらに語彙を制限した超小規模タスク用にフィルタリング
    # これにより、モデル容量と問題との関係を見る
    filtered_data = []
    for item in micro_data:
        filtered_text = ' '.join([w for w in item['text'].split() 
                                if len(filtered_data) < size])
        if filtered_text:
            filtered_data.append({'text': filtered_text})
    
    return micro_data

# 追加: 複素数初期化関数
def initialize_complex_parameters(model, scale=0.02):
    """複素数パラメータの特殊初期化"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'embedding' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'norm' in name or 'scale' in name:
                nn.init.constant_(param, 1.0)
            elif 'wave' in name or 'complex' in name:
                # 複素数パラメータ用の特殊初期化
                # 絶対値と位相を分けて初期化
                with torch.no_grad():
                    # 実部と虚部を個別に初期化
                    if param.dim() >= 2:
                        # 複素数行列の特殊初期化
                        nn.init.uniform_(param, -scale, scale)
                        
                        # 位相をより均一に分布させる工夫
                        if 'phase' in name:
                            nn.init.uniform_(param, -np.pi, np.pi)
                    else:
                        nn.init.zeros_(param)
            elif 'classifier' in name:
                # 出力層は非常に小さく初期化
                nn.init.normal_(param, mean=0.0, std=0.001)
            else:
                # 一般的な重み行列はXavier初期化
                nn.init.xavier_uniform_(param, gain=0.1)
        elif 'bias' in name:
            nn.init.zeros_(param)

# 追加: 勾配フロー解析関数
def analyze_gradient_flow(model, input_ids, labels):
    """モデル全体の勾配フローを解析"""
    # 計算グラフの追跡を有効化
    for param in model.parameters():
        if param.requires_grad:
            param.retain_grad()
    
    # 順伝播と損失計算
    outputs = model(input_ids)
    loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
    loss.backward()
    
    # 勾配フロー解析
    gradient_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad.data
            grad_stats = {
                'mean': float(grad.mean()),
                'std': float(grad.std()),
                'min': float(grad.min()),
                'max': float(grad.max()),
                'norm': float(torch.norm(grad)),
                'has_nan': bool(torch.isnan(grad).any()),
                'has_inf': bool(torch.isinf(grad).any())
            }
            gradient_stats[name] = grad_stats
    
    return gradient_stats, float(loss.item())

# 追加: 生成テキスト品質評価
def evaluate_generated_text(model, tokenizer, input_text, max_len=50, device="cuda"):
    """モデルの現在の状態を生成テキストで評価"""
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    generated = sample_text(model, input_ids, max_len=max_len, device=device)
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text

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
    
    # モデル設定: 複数のモデルサイズを試す
    model_configs = [
        # 超小型モデル（問題切り分け用）
        ModelConfig(
            hidden_size=64,     # 非常に小さいモデルサイズ
            num_layers=1,       # 最少レイヤー数
            max_seq_len=32,     # 非常に短いシーケンス
            dropout_prob=0.0,   # ドロップアウトなし
            use_rope=True,      # 位置エンコーディングは必須
            norm_scheme="pre"   # Pre-LN方式を試す
        ),
        # 小型モデル
        ModelConfig(
            hidden_size=256,    # 小さいモデルサイズ
            num_layers=2,       # 少ないレイヤー数
            max_seq_len=128,    # 短いシーケンス
            dropout_prob=0.0,   # まずドロップアウトをなしで試す
            use_rope=True,      # 位置エンコーディングは必須
            norm_scheme="post"  # Post-LN方式も試す
        ),
    ]
    
    # 複数のトレーニング設定を試す
    training_configs = [
        # 安定性重視の設定
        TrainingConfig(
            learning_rate=5e-5,     # 小さな学習率
            batch_size=8,           # 小さなバッチサイズ
            mlm_epochs=15,          # より多くのエポック
            mlm_probability=0.15,   # 標準的なマスク率
            weight_decay=0.01,      # 適度なウェイト減衰
            warmup_steps=500,       # 長めのウォームアップ
            accumulation_steps=4,   # 勾配累積
            use_amp=False,          # AMP無効（数値安定性のため）
            clip_value=1.0,         # 強い勾配クリッピング
        ),
        # 探索的な設定
        TrainingConfig(
            learning_rate=2e-4,     # 大きめの学習率
            batch_size=16,          # 標準的なバッチサイズ
            mlm_epochs=10,          # 標準的なエポック数
            mlm_probability=0.15,   # 標準的なマスク率
            weight_decay=0.0,       # ウェイト減衰なし
            warmup_steps=100,       # 短いウォームアップ
            accumulation_steps=1,   # 勾配累積なし
            use_amp=False,          # AMP無効
            clip_value=5.0,         # 緩い勾配クリッピング
        ),
    ]
    
    # ディレクトリ準備
    deep_fix_dir = os.path.join(paths_config.checkpoint_dir, "deep_fix_model")
    os.makedirs(deep_fix_dir, exist_ok=True)
    os.makedirs(os.path.join(paths_config.log_dir, "wave_debug"), exist_ok=True)
    
    try:
        if device.type != "cuda":
            print("このスクリプトはGPU環境で実行してください。")
            sys.exit(1)
        
        print("=== 抜本修正版学習を実行します ===")
        
        # データセット読み込み
        train_dataset = load_from_disk(os.path.join(paths_config.data_dir, "train_dataset"))
        valid_dataset = load_from_disk(os.path.join(paths_config.data_dir, "valid_dataset"))
        
        print(f"元のデータセットサイズ: {len(train_dataset)}")
        
        # 段階的なデータセット切り分け
        dataset_sizes = [100, 1000, 10000]
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(paths_config.tokenizer_name)
        
        # 各モデル構成で実験
        for model_idx, model_config in enumerate(model_configs):
            model_config.set_tokenizer(tokenizer)
            print(f"\n=== モデル構成 {model_idx+1}/{len(model_configs)} を検証 ===")
            print(f"hidden_size: {model_config.hidden_size}, layers: {model_config.num_layers}, norm: {model_config.norm_scheme}")
            
            # 各データセットサイズで実験
            for dataset_size in dataset_sizes:
                print(f"\n--- データセットサイズ: {dataset_size} ---")
                
                # データセット準備
                train_dataset_small = train_dataset.select(range(min(dataset_size, len(train_dataset))))
                valid_dataset_small = valid_dataset.select(range(min(dataset_size//10, len(valid_dataset))))
                
                # 超小規模テスト用のデータセットも準備
                micro_dataset = create_micro_dataset(train_dataset, size=100)
                
                # モデル初期化
                model = WaveNetworkLM(model_config)
                
                # 複素数パラメータの特別初期化
                initialize_complex_parameters(model, scale=0.01)
                
                # 各トレーニング設定で実験
                for train_idx, training_config in enumerate(training_configs):
                    print(f"\n-- トレーニング設定 {train_idx+1}/{len(training_configs)} --")
                    print(f"学習率: {training_config.learning_rate}, バッチサイズ: {training_config.batch_size}")
                    
                    # モデルをデバイスに移動
                    model.to(device)
                    
                    # トレーナー初期化
                    trainer = Trainer(
                        model=model,
                        train_dataset=train_dataset_small,
                        valid_dataset=valid_dataset_small,
                        training_config=training_config,
                        device=device,
                        paths_config=paths_config
                    )
                    
                    # 勾配クリッピング設定
                    trainer.clip_value = training_config.clip_value
                    
                    # 事前分析: 勾配フロー
                    print("初期状態での勾配フロー分析...")
                    collator = CustomCollator(
                        tokenizer=tokenizer,
                        model_config=model_config,
                        mlm=True,
                        mlm_probability=training_config.mlm_probability,
                        mask_token_id=tokenizer.mask_token_id,
                        qa=False
                    )
                    
                    dataloader = DataLoader(
                        train_dataset_small.select(range(min(5, len(train_dataset_small)))),
                        batch_size=1,
                        shuffle=False,
                        collate_fn=collator
                    )
                    
                    batch = next(iter(dataloader))
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    
                    grad_stats, loss = analyze_gradient_flow(model, input_ids, labels)
                    
                    # 勾配問題の検出
                    has_grad_problems = False
                    for name, stats in grad_stats.items():
                        if stats['has_nan'] or stats['has_inf'] or stats['norm'] > 10.0:
                            has_grad_problems = True
                            print(f"  警告: {name}の勾配に問題があります。norm={stats['norm']:.4f}")
                    
                    # 問題がある場合はスケールダウン
                    if has_grad_problems:
                        print("  勾配に問題があるため、初期学習率を0.1倍に調整します")
                        for param_group in trainer.optimizer.param_groups:
                            param_group['lr'] *= 0.1
                    
                    # 波表現の診断
                    print("波表現の診断...")
                    wave_stats = WaveDebugger.check_wave_representation(
                        model, 
                        input_ids, 
                        save_path=os.path.join(paths_config.log_dir, "wave_debug", f"wave_repr_m{model_idx}_t{train_idx}_d{dataset_size}.png")
                    )
                    
                    # 生成テキスト評価（学習前）
                    print("学習前の生成テキスト評価:")
                    prompt = "これは"
                    generated = evaluate_generated_text(model, tokenizer, prompt, max_len=20, device=device)
                    print(f"プロンプト「{prompt}」→ 「{generated}」")
                    
                    # 学習
                    print("学習開始...")
                    history = trainer.train_mlm(num_epochs=training_config.mlm_epochs)
                    
                    # 結果評価
                    print("学習結果:")
                    min_loss = min([log['loss'] for log in history])
                    print(f"  最小ロス: {min_loss:.4f}")
                    
                    # 生成テキスト評価（学習後）
                    print("学習後の生成テキスト評価:")
                    generated = evaluate_generated_text(model, tokenizer, prompt, max_len=20, device=device)
                    print(f"プロンプト「{prompt}」→ 「{generated}」")
                    
                    # チェックポイント保存（小さなデータセットの場合は省略）
                    if dataset_size >= 1000:
                        checkpoint_path = f"deep_fix_model/model_m{model_idx}_t{train_idx}_d{dataset_size}"
                        trainer.save_checkpoint(checkpoint_path)
                        print(f"モデルを保存しました: {os.path.join(deep_fix_dir, checkpoint_path)}.pt")
                    
                    # 勾配フロー再分析（学習後）
                    grad_stats_after, loss_after = analyze_gradient_flow(model, input_ids, labels)
                    print(f"学習後の損失: {loss_after:.4f} (初期値: {loss:.4f})")
                    
                    # 勾配変化の分析
                    for name in grad_stats.keys():
                        before = grad_stats[name]['norm']
                        after = grad_stats_after[name]['norm']
                        if after / (before + 1e-10) > 10 or after / (before + 1e-10) < 0.1:
                            print(f"  層 {name} の勾配が大きく変化: {before:.4f} → {after:.4f}")
                    
                    # トレーナークリーンアップ
                    trainer.close()
        
        # 最終的な総括
        print("\n=== 実験総括 ===")
        print("複数のモデル構成、データセットサイズ、トレーニング設定を試した結果:")
        print("1. データセットサイズの影響: 小→大でどのように変化したか")
        print("2. モデル構成の影響: Pre-LN vs Post-LN、サイズによる違い")
        print("3. トレーニング設定の影響: 学習率、勾配クリッピング等の効果")
        
        # 最適な設定でフルトレーニング
        print("\n=== 最良設定でのフルトレーニング ===")
        # ここでは最適な設定を選んでフルトレーニングを行う
        # 実際には上記の実験結果から選ぶべき
        
        best_model_config = model_configs[-1]  # 最大のモデルを選択
        best_training_config = training_configs[0]  # 安定性重視の設定を選択
        
        print(f"選択したモデル構成: hidden_size={best_model_config.hidden_size}, layers={best_model_config.num_layers}")
        print(f"選択したトレーニング設定: lr={best_training_config.learning_rate}, batch={best_training_config.batch_size}")
        
        # データセット準備（より大きなサイズ）
        train_dataset_full = train_dataset.select(range(min(100000, len(train_dataset))))
        valid_dataset_full = valid_dataset.select(range(min(5000, len(valid_dataset))))
        print(f"トレーニングデータサイズ: {len(train_dataset_full)}")
        
        # 最終モデル初期化
        best_model_config.set_tokenizer(tokenizer)
        final_model = WaveNetworkLM(best_model_config)
        initialize_complex_parameters(final_model, scale=0.01)
        final_model.to(device)
        
        # 最終トレーナー
        final_trainer = Trainer(
            model=final_model,
            train_dataset=train_dataset_full,
            valid_dataset=valid_dataset_full,
            training_config=best_training_config,
            device=device,
            paths_config=paths_config
        )
        
        # 最終トレーニング
        print("最終トレーニング開始...")
        final_history = final_trainer.train_mlm(num_epochs=best_training_config.mlm_epochs)
        
        # 結果評価と保存
        min_loss = min([log['loss'] for log in final_history])
        print(f"最終最小ロス: {min_loss:.4f}")
        
        final_trainer.save_checkpoint("deep_fix_model/final_model")
        print(f"最終モデルを保存しました: {os.path.join(deep_fix_dir, 'final_model.pt')}")
        
        final_trainer.close()
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'final_trainer' in locals():
            final_trainer.close()
        
        print(f"=== 処理完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

if __name__ == "__main__":
    main()
