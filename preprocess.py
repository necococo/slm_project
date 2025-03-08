#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
英語データセット（WikiText-2）の前処理スクリプト
データセットのダウンロード、トークン化、保存を行います
"""

import os
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from slm.config import PathsConfig

def parse_arguments():
    """コマンドライン引数を解析します"""
    parser = argparse.ArgumentParser(description='英語データセット前処理スクリプト')
    parser.add_argument('--language', type=str, default='en', choices=['en', 'ja'],
                       help='処理する言語（en: 英語, ja: 日本語）')
    parser.add_argument('--chunk_size', type=int, default=1000,
                       help='一度に処理するチャンクサイズ')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='処理済みデータの保存先（デフォルト: 自動設定）')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='データ処理に使用するワーカー数（並列処理）')
    return parser.parse_args()

def setup_paths_and_tokenizer(language='en'):
    """パス設定とトークナイザーの初期化を行います"""
    # パス設定
    paths_config = PathsConfig(language=language)
    
    # ディレクトリ作成
    os.makedirs(paths_config.data_dir, exist_ok=True)
    
    # トークナイザー読み込み
    print(f"トークナイザーをロード中: {paths_config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(paths_config.tokenizer_name)
    
    # GPT-2トークナイザーの場合、パディングトークンが設定されていないため追加
    if paths_config.tokenizer_name == "gpt2" and tokenizer.pad_token is None:
        print("GPT-2トークナイザーにパディングトークンを設定します")
        # EOS（文末）トークンをパディングトークンとしても使用
        tokenizer.pad_token = tokenizer.eos_token
        # または新しいパディングトークンを追加する場合:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    return paths_config, tokenizer

def download_and_prepare_dataset(paths_config):
    """データセットをダウンロードします（サイズ調整なし）"""
    print(f"データセットをダウンロード中: {paths_config.dataset_name}/{paths_config.dataset_subset}")
    
    try:
        dataset = load_dataset(paths_config.dataset_name, paths_config.dataset_subset)
        print(f"データセットロード完了:")
        for split, ds in dataset.items():
            print(f"  - {split}: {len(ds)}件")
        
        return dataset
    
    except Exception as e:
        print(f"データセットのダウンロード中にエラーが発生しました: {e}")
        raise

def calculate_optimal_sequence_length(dataset, tokenizer, sample_size=1000):
    """データセットの適切な最大シーケンス長を計算します"""
    print("データセットの最適なシーケンス長を計算しています...")
    
    # サンプリング対象の件数を制限
    sample_size = min(sample_size, len(dataset['train']))
    
    # ランダムサンプリング
    import numpy as np
    indices = np.random.choice(len(dataset['train']), sample_size, replace=False)
    samples = [dataset['train'][int(i)]['text'] for i in indices]
    
    # 各サンプルのトークン長を計算
    lengths = []
    for text in tqdm(samples, desc="シーケンス長計算"):
        tokens = tokenizer.encode(text)
        lengths.append(len(tokens))
    
    # 統計情報を計算
    avg_length = sum(lengths) / len(lengths)
    max_length = max(lengths)
    p95_length = sorted(lengths)[int(len(lengths) * 0.95)]  # 95パーセンタイル
    
    print(f"シーケンス長の統計:")
    print(f"  - 平均: {avg_length:.1f} トークン")
    print(f"  - 最大: {max_length} トークン")
    print(f"  - 95パーセンタイル: {p95_length} トークン")
    
    # 最大長を返す（必要に応じて95パーセンタイルを使用）
    # 極端に長いテキストがある場合は 95 パーセンタイルの方が適切
    suggested_length = p95_length
    
    print(f"推奨シーケンス長: {suggested_length}")
    return suggested_length

def tokenize_dataset(dataset, tokenizer, batch_size=64, chunk_size=1000, num_workers=4):
    """データセットをトークン化します"""
    def tokenize_function(examples):
        # トークナイザーがパディングトークンを持っているか確認
        if tokenizer.pad_token is None:
            raise ValueError("トークナイザーにパディングトークンが設定されていません")
            
        outputs = tokenizer(
            examples["text"],
            truncation=False,  # 切り詰めを行わない
            padding=False,     # パディングも行わない（Collatorに任せる）
            return_tensors=None
        )
        return outputs
    
    print("データセットをトークン化しています...")
    print("注意: 切り詰めやパディングは行わず、元のテキスト長を維持します。トレーニング時にCollatorが処理します。")
    print(f"並列ワーカー数: {num_workers}")
    
    # 大きなデータセットを効率的に処理するためにチャンク化
    tokenized_dataset = {}
    
    for split, ds in dataset.items():
        print(f"  - {split}セットを処理中...")
        
        # データセットが大きい場合はチャンク単位で処理
        if len(ds) > chunk_size:
            tokenized_chunks = []
            for i in tqdm(range(0, len(ds), chunk_size), desc=f"{split} tokenization"):
                end_idx = min(i + chunk_size, len(ds))
                chunk = ds.select(range(i, end_idx))
                tokenized_chunk = chunk.map(
                    tokenize_function,
                    batched=True,
                    batch_size=batch_size,
                    remove_columns=["text"],
                    num_proc=num_workers  # 並列処理でCPU効率を向上
                )
                tokenized_chunks.append(tokenized_chunk)
            
            # チャンクを連結
            from datasets import concatenate_datasets
            tokenized_dataset[split] = concatenate_datasets(tokenized_chunks)
        else:
            # 小さいデータセットは一度に処理
            tokenized_dataset[split] = ds.map(
                tokenize_function,
                batched=True,
                batch_size=batch_size,
                remove_columns=["text"],
                num_proc=num_workers  # 並列処理でCPU効率を向上
            )
    
    # サンプルを表示
    print("\nトークン化サンプル:")
    for split in tokenized_dataset:
        print(f"\n{split}セットの例:")
        sample = tokenized_dataset[split][0]
        print(f"input_ids: {sample['input_ids'][:10]}... (長さ: {len(sample['input_ids'])})")
        print(f"attention_mask: {sample['attention_mask'][:10]}... (長さ: {len(sample['attention_mask'])})")
        # デコードしてオリジナルテキストを復元
        decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        print(f"デコード結果: {decoded[:50]}...")
    
    return tokenized_dataset

def save_processed_dataset(tokenized_dataset, save_dir):
    """処理済みデータセットを保存します"""
    # 親ディレクトリを含めて再帰的に作成
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nデータセットを保存中: {save_dir}")
    
    for split, ds in tokenized_dataset.items():
        split_dir = os.path.join(save_dir, split)
        ds.save_to_disk(split_dir)
        print(f"  - {split}セットを保存しました: {split_dir}")
        
    # 保存した場所へのフルパスを表示
    abs_path = os.path.abspath(save_dir)
    print(f"\n保存完了: {abs_path}")

def main():
    """メイン実行関数"""
    # 引数解析
    args = parse_arguments()
    
    print(f"=== 英語データセット前処理処理 - {args.language} ===")
    
    # GPUの有無を確認し、情報を表示
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"GPU検出: {torch.cuda.get_device_name(0)}")
        print("注意: 前処理（特にトークン化）はGPUを活用するタスクではありません。")
        print("      代わりにCPU並列処理を使用します。GPUはモデル学習フェーズで活用されます。")
    else:
        print("GPUは検出されませんでした。CPUで処理を続行します。")
    
    # パス設定とトークナイザー初期化
    paths_config, tokenizer = setup_paths_and_tokenizer(args.language)
    
    # データセットのダウンロードと準備（サイズ調整なし）
    dataset = download_and_prepare_dataset(paths_config)
    
    # 最大シーケンス長の設定 (統計情報収集のみ使用)
    max_length = args.max_length
    if args.auto_max_length:
        max_length = calculate_optimal_sequence_length(dataset, tokenizer)
        print(f"自動計算したシーケンス長: {max_length} (統計情報のみ、切り詰めは行いません)")
    else:
        print(f"注意: 指定されたmax_length={max_length}は統計情報のみに使用され、切り詰めは行いません")
    
    # トークン化 (サイズ調整なし)
    tokenized_dataset = tokenize_dataset(
        dataset, 
        tokenizer,
        batch_size=64,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers
    )
    
    # シーケンス長の統計情報を収集
    lengths = []
    for split, ds in tokenized_dataset.items():
        split_lengths = [len(example['input_ids']) for example in ds]
        print(f"{split}データの長さ統計:")
        print(f"  - 最小: {min(split_lengths)} トークン")
        print(f"  - 最大: {max(split_lengths)} トークン")
        print(f"  - 平均: {sum(split_lengths)/len(split_lengths):.1f} トークン")
        print(f"  - 95パーセンタイル: {sorted(split_lengths)[int(len(split_lengths) * 0.95)]} トークン")
        lengths.extend(split_lengths)
    
    # 保存先設定
    if args.save_dir:
        save_dir = args.save_dir
    else:
        # デフォルトの保存先（オリジナル長を明示）
        save_dir = os.path.join(paths_config.data_dir, "processed_raw")
    
    # 処理済みデータセット保存
    save_processed_dataset(tokenized_dataset, save_dir)
    
    # 次のステップの指示
    print("\n=== 前処理完了 ===")
    print(f"\n次のステップ: 以下のコマンドで様々なサイズ・シーケンス長でトレーニングを実行できます:")
    print(f"python training.py --data_dir {save_dir} --sequence_length 128 --train_samples 500")
    print(f"python training.py --data_dir {save_dir} --sequence_length 256 --train_samples 1000")
    print(f"python training.py --data_dir {save_dir} --sequence_length 512 --train_samples 5000")
    print(f"python training.py --data_dir {save_dir} --sequence_length 128 --sample_ratio 0.1")

if __name__ == "__main__":
    main()
