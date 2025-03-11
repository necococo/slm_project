#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
toramaru-u/wiki40b-ja データセットをロードして確認するスクリプト
このデータセットは日本語Wikipediaのテキストから特殊トークンやタイトル削除済み
"""

import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slm.tokenizer import JapaneseTokenizer

def load_and_check():
    """データセットをロードして内容を確認する"""
    print("toramaru-u/wiki40b-ja データセットをロード中...")
    
    # データセットをロード
    dataset = load_dataset("toramaru-u/wiki40b-ja")
    
    # データセット情報
    print(f"\nデータセット構造:")
    print(f"  スプリット: {list(dataset.keys())}")
    for split in dataset:
        print(f"  {split}: {len(dataset[split])}サンプル")
    
    # 各スプリットのカラム名
    print(f"\nデータセットのカラム:")
    for split in dataset:
        print(f"  {split}: {list(dataset[split].features)}")
    
    # サンプルデータの表示
    print("\nサンプルデータ:")
    for split in dataset:
        sample = dataset[split][0]
        print(f"\n{split}サンプル:")
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")
    
    return dataset

def test_with_tokenizer(dataset):
    """トークナイザーでテキストをエンコード・デコードしてみる"""
    print("\n=== megagonlabs/t5-base-japanese-web トークナイザーでテスト ===")
    
    # トークナイザーをロード
    hf_tokenizer = AutoTokenizer.from_pretrained("megagonlabs/t5-base-japanese-web")
    jp_tokenizer = JapaneseTokenizer.from_pretrained_tokenizer(hf_tokenizer)
    
    # 訓練データから数サンプルを選択
    samples = []
    if 'train' in dataset:
        samples.extend([dataset['train'][i]['text'] for i in range(min(2, len(dataset['train'])))])
    if 'validation' in dataset:
        samples.extend([dataset['validation'][i]['text'] for i in range(min(1, len(dataset['validation'])))])
    
    # サンプルテキストが空の場合のフォールバック
    if not samples:
        samples = ["これはWikipediaのテストテキストです。"]
    
    # 各サンプルでトークン化テスト
    for i, text in enumerate(samples):
        print(f"\nサンプル {i+1}:")
        sample_text = text[:100]  # 最初の100文字だけ使用
        print(f"テキスト: {sample_text}")
        
        # トークン化
        token_ids = jp_tokenizer.encode(sample_text)
        tokens = hf_tokenizer.convert_ids_to_tokens(token_ids)
        print(f"トークンID: {token_ids}")
        print(f"トークン: {tokens}")
        
        # デコード
        decoded = jp_tokenizer.decode(token_ids)
        print(f"デコード結果: {decoded}")
        
        # 一致確認
        if sample_text == decoded:
            print("✓ 完全一致")
        else:
            print("× 不一致")
            print(f"  元のテキスト: {sample_text}")
            print(f"  デコード結果: {decoded}")

def save_to_local(dataset, output_dir):
    """データセットをローカルに保存する"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nデータセットをローカルに保存: {output_dir}")
    dataset.save_to_disk(output_dir)
    print(f"保存完了！")

if __name__ == "__main__":
    # コマンドライン引数
    import argparse
    parser = argparse.ArgumentParser(description='toramaru-u/wiki40b-ja データセットの確認とテスト')
    parser.add_argument('--save', action='store_true', help='データセットをローカルに保存する')
    parser.add_argument('--output-dir', type=str, default='./data/wiki40b_ja', 
                        help='データセットの保存先ディレクトリ')
    args = parser.parse_args()
    
    # データセットをロードして確認
    dataset = load_and_check()
    
    # トークナイザーでテスト
    test_with_tokenizer(dataset)
    
    # 保存オプションが指定されている場合
    if args.save:
        save_to_local(dataset, args.output_dir)