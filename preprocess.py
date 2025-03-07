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
    parser.add_argument('--max_samples', type=int, default=None,
                       help='処理するサンプル数の上限（デフォルト: すべて）')
    parser.add_argument('--max_length', type=int, default=128,
                       help='トークン化する最大シーケンス長')
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
    
    return paths_config, tokenizer

def download_and_prepare_dataset(paths_config, max_samples=None):
    """データセットをダウンロードして基本的な前処理を行います"""
    print(f"データセットをダウンロード中: {paths_config.dataset_name}/{paths_config.dataset_subset}")
    
    try:
        dataset = load_dataset(paths_config.dataset_name, paths_config.dataset_subset)
        print(f"データセットロード完了:")
        for split, ds in dataset.items():
            print(f"  - {split}: {len(ds)}件")
        
        # サンプル数を制限
        if max_samples:
            for split in dataset:
                if len(dataset[split]) > max_samples:
                    if split == 'train':
                        dataset[split] = dataset[split].select(range(max_samples))
                    else:
                        # 検証・テストセットはトレーニングセットの1/10のサイズに
                        subset_size = max(1, max_samples // 10)
                        dataset[split] = dataset[split].select(range(min(subset_size, len(dataset[split]))))
            
            print(f"サンプル数を制限しました:")
            for split, ds in dataset.items():
                print(f"  - {split}: {len(ds)}件")
        
        return dataset
    
    except Exception as e:
        print(f"データセットのダウンロード中にエラーが発生しました: {e}")
        raise

def tokenize_dataset(dataset, tokenizer, max_length=128, batch_size=64, chunk_size=1000, num_workers=4):
    """データセットをトークン化します"""
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        return outputs
    
    print("データセットをトークン化しています...")
    print(f"注意: トークン化はCPU処理に最適化されたタスクです。GPUは使用しません。")
    print(f"      並列ワーカー数: {num_workers}")
    
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
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nデータセットを保存中: {save_dir}")
    
    for split, ds in tokenized_dataset.items():
        split_dir = os.path.join(save_dir, split)
        ds.save_to_disk(split_dir)
        print(f"  - {split}セットを保存しました: {split_dir}")

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
    
    # データセットのダウンロードと準備
    dataset = download_and_prepare_dataset(paths_config, args.max_samples)
    
    # トークン化 (CPUベースの並列処理を使用)
    tokenized_dataset = tokenize_dataset(
        dataset, 
        tokenizer, 
        max_length=args.max_length,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers
    )
    
    # 保存先設定
    if args.save_dir:
        save_dir = args.save_dir
    else:
        # デフォルトの保存先
        save_dir = os.path.join(paths_config.data_dir, "processed")
    
    # 処理済みデータセット保存
    save_processed_dataset(tokenized_dataset, save_dir)
    
    # 次のステップの指示
    print("\n=== 前処理完了 ===")
    print(f"次のステップ: 以下のコマンドでトレーニングを実行できます")
    print(f"python training.py --data_dir {save_dir}")

if __name__ == "__main__":
    main()
