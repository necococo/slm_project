#!/usr/bin/env python
# coding: utf-8
"""
Wiki40B-ja用テストデータセットを準備するツール
生データをダウンロードしてトークン化し、テスト用に保存します
"""

import os
import sys
import argparse
import time
import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# 必要なモジュールへのパスを追加
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from slm.tokenizer import JapaneseTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Wiki40B-jaテストデータセットを準備する")
    
    # データパス関連
    parser.add_argument("--dataset_name", type=str, default="toramaru-u/wiki40b-ja",
                        help="Hugging Faceからロードするデータセット名")
    parser.add_argument("--output_dir", type=str, default="./data/wiki40b_ja_test",
                        help="処理済みデータセットの保存先ディレクトリ")
    parser.add_argument("--tokenizer_name", type=str, default="megagonlabs/t5-base-japanese-web",
                        help="使用するトークナイザー名")
    
    # 処理設定
    parser.add_argument("--test_samples", type=int, default=1000,
                        help="テストデータセットのサンプル数")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="最大シーケンス長")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="処理時のバッチサイズ")
    parser.add_argument("--seed", type=int, default=42,
                        help="ランダムシード")
    
    return parser.parse_args()


def load_tokenizer(tokenizer_name):
    """トークナイザーをロードする関数"""
    print(f"トークナイザー {tokenizer_name} をロード中...")
    
    # まずHuggingFaceからロード
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # T5トークナイザーにはデフォルトのマスクトークンがないため、必要に応じて追加
    if not hasattr(hf_tokenizer, 'mask_token') or hf_tokenizer.mask_token is None:
        # マスクトークンの追加
        hf_tokenizer.add_special_tokens({'mask_token': '<mask>'})
        print(f"マスクトークン '<mask>' を追加しました。")
    
    # BOSトークンの追加（必要に応じて）
    if not hasattr(hf_tokenizer, 'bos_token') or hf_tokenizer.bos_token is None:
        hf_tokenizer.add_special_tokens({'bos_token': '<s>'})
        print(f"BOSトークン '<s>' を追加しました。")
    
    # JapaneseTokenizerラッパーに変換
    jp_tokenizer = JapaneseTokenizer.from_pretrained_tokenizer(hf_tokenizer)
    
    print(f"トークナイザーをロードしました。語彙サイズ: {len(hf_tokenizer.vocab) if hasattr(hf_tokenizer, 'vocab') else hf_tokenizer.vocab_size}")
    
    return jp_tokenizer, hf_tokenizer


def clean_text(batch):
    """テキスト内のプレースホルダートークンを削除する関数"""
    text = batch["text"]
    for token in ["_START_ARTICLE_", "_START_SECTION_", "_START_PARAGRAPH_"]:
        text = text.replace(token, "")
    text = text.replace("_NEWLINE_", "\n")
    batch["text"] = text
    return batch


def tokenize_function(examples, tokenizer, max_seq_len):
    """テキストをトークン化する関数"""
    tokenized = {"input_ids": [], "attention_mask": []}
    
    for text in examples["text"]:
        # トークン化 - HuggingFaceトークナイザーを使用
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        
        # 最大長に切り詰め
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
        
        # 注意マスクを作成（すべて1）
        attn_mask = [1] * len(token_ids)
        
        tokenized["input_ids"].append(token_ids)
        tokenized["attention_mask"].append(attn_mask)
    
    return tokenized


def prepare_dataset(args):
    """データセットを準備する関数"""
    print(f"Hugging Faceからデータセット {args.dataset_name} をロード中...")
    dataset = load_dataset(args.dataset_name)
    
    print("データセット情報:")
    for split in dataset:
        print(f"  {split}: {len(dataset[split])}サンプル")
    
    # トークナイザーのロード
    _, hf_tokenizer = load_tokenizer(args.tokenizer_name)
    
    # テストデータセットを準備（testが存在しない場合はvalidationを使用）
    if "test" in dataset:
        test_split = "test"
    elif "validation" in dataset:
        test_split = "validation"
    else:
        print("警告: テストもバリデーションも見つかりません。トレインデータを使用します。")
        test_split = "train"
    
    # データサイズの制限
    if len(dataset[test_split]) > args.test_samples:
        print(f"テストデータを{args.test_samples}サンプルに制限します")
        # シード固定でランダムサンプリング
        random_indices = torch.randperm(len(dataset[test_split]), generator=torch.Generator().manual_seed(args.seed))[:args.test_samples]
        test_dataset = dataset[test_split].select(random_indices.tolist())
    else:
        test_dataset = dataset[test_split]
    
    print(f"選択されたテストデータ: {len(test_dataset)}サンプル")
    
    # プレーンテキスト形式のテストデータを保存
    plain_text_path = os.path.join(args.output_dir, "wiki40b_ja_test_plain.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"プレーンテキストデータを保存中: {plain_text_path}")
    with open(plain_text_path, "w", encoding="utf-8") as f:
        for item in tqdm(test_dataset, desc="テキスト保存中"):
            # プレースホルダーの置き換え
            text = clean_text({"text": item["text"]})["text"]
            f.write(text + "\n\n")
    
    # テキストの前処理
    print("テキストデータをクリーニング中...")
    cleaned_dataset = test_dataset.map(clean_text)
    
    # データセットのトークン化
    print(f"データセットをトークン化中...")
    batch_size = args.batch_size
    
    tokenized_dataset = cleaned_dataset.map(
        lambda examples: tokenize_function(examples, hf_tokenizer, args.max_seq_len),
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"],
        desc="トークン化中"
    )
    
    # テストデータセットを保存
    tokenized_dataset_path = os.path.join(args.output_dir, "tokenized_dataset")
    print(f"トークン化済みデータセットを保存中: {tokenized_dataset_path}")
    
    # 直接データセットを保存
    tokenized_dataset.save_to_disk(tokenized_dataset_path)
    
    print("データセットのサンプルを表示:")
    if len(tokenized_dataset) > 0:
        sample = tokenized_dataset[0]
        print(f"サンプルの形状: input_ids={len(sample['input_ids'])}")
        print(f"最初の20トークン: {sample['input_ids'][:20]}")
        decoded_text = hf_tokenizer.decode(sample['input_ids'])
        print(f"デコード結果: {decoded_text[:100]}..." if len(decoded_text) > 100 else decoded_text)
    
    return tokenized_dataset_path


def main():
    args = parse_args()
    
    # 乱数シード設定
    torch.manual_seed(args.seed)
    
    # 処理時間計測開始
    start_time = time.time()
    
    # テストデータセットの準備
    test_dataset_path = prepare_dataset(args)
    
    # 処理時間計測終了
    processing_time = time.time() - start_time
    print(f"\nテストデータセットの準備が完了しました: {test_dataset_path}")
    print(f"処理時間: {processing_time:.2f}秒")
    
    # 使用方法の説明
    print("\n使用方法:")
    print(f"  python -m slm.tools.test_wiki40b_ja_model --model_path=PATH_TO_MODEL --use_local_dataset --local_data_dir={test_dataset_path}")


if __name__ == "__main__":
    main()