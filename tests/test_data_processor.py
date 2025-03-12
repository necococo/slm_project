#!/usr/bin/env python
# coding: utf-8
# tests/test_data_processor.py
# データプロセッサーのテスト用スクリプト

import os
import sys
import argparse
import torch
# slmモジュールからインポートするよう修正
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slm.data_processor import SimpleDataProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="データプロセッサーのテスト")
    
    parser.add_argument("--tokenizer_name", type=str, default="megagonlabs/t5-base-japanese-web",
                        help="使用するトークナイザー名")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大シーケンス長")
    parser.add_argument("--mask_token", type=str, default="<mask>",
                        help="マスクトークン")
    parser.add_argument("--mask_ratio", type=float, default=0.2,
                        help="マスク割合")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="単一データセットのパス (非推奨)")
    parser.add_argument("--data_dir", type=str, default="/content/drive/MyDrive/slm/data/fujiki",
                        help="データディレクトリ (train/valid/testを含む)")
    parser.add_argument("--sample_all_splits", action="store_true",
                        help="train/valid/testすべてからサンプリングする")
    parser.add_argument("--samples_per_split", type=int, default=1,
                        help="各分割から抽出するサンプル数")
    parser.add_argument("--test_text", type=str, default=None,
                        help="テスト用テキスト（指定時はデータセットより優先）")
    
    return parser.parse_args()

def get_sample_from_dataset(processor, dataset_path, split_name=""):
    """指定したデータセットから1つのサンプルを取得する"""
    try:
        # データセットのロード
        dataset = processor.load_dataset(dataset_path)
        dataset_size = len(dataset)
        
        if dataset_size == 0:
            print(f"警告: {split_name}データセットは空です")
            return None
        
        # ランダムにサンプルを選択
        import random
        random_idx = random.randint(0, dataset_size - 1)
        random_sample = dataset[random_idx]
        
        # サンプル情報
        print(f"\n=== {split_name}データセットから選択したサンプル（インデックス: {random_idx}/{dataset_size-1}） ===")
        
        # サンプルからテキストを抽出
        if "text" in random_sample:
            sample_text = random_sample["text"]
            print(f"テキスト（プレビュー）: {sample_text[:100]}..." if len(sample_text) > 100 else sample_text)
            return {"text": sample_text, "source": "text"}
        elif "input_ids" in random_sample:
            # トークンIDからテキストを復元
            input_ids = random_sample["input_ids"]
            sample_text = processor.tokenizer.decode(input_ids)
            print(f"デコードテキスト（プレビュー）: {sample_text[:100]}..." if len(sample_text) > 100 else sample_text)
            return {"text": sample_text, "source": "input_ids", "input_ids": input_ids}
        else:
            print(f"警告: {split_name}データセットの選択したサンプルに'text'または'input_ids'フィールドがありません")
            print(f"サンプルのキー: {list(random_sample.keys())}")
            return None
    except Exception as e:
        print(f"{split_name}データセットのロード中にエラーが発生しました: {e}")
        return None

def main():
    # コマンドライン引数の解析
    args = parse_args()
    
    # データプロセッサーの初期化
    print("データプロセッサーを初期化中...")
    processor = SimpleDataProcessor(
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
        mask_token=args.mask_token
    )
    
    print("\n=== マスキングとデコードのテスト ===")
    print(f"マスク割合: {args.mask_ratio:.2f}")
    
    # サンプルのリスト
    samples = []
    
    # 引数でテキストが直接指定されている場合
    if args.test_text:
        samples.append({"text": args.test_text, "source": "argument"})
        print(f"\n=== テスト用テキスト（引数から） ===")
        print(f"テキスト（プレビュー）: {args.test_text[:100]}..." if len(args.test_text) > 100 else args.test_text)
    # すべての分割からサンプリングする場合
    elif args.sample_all_splits and args.data_dir:
        print(f"\n=== すべての分割からサンプリング (各{args.samples_per_split}件) ===")
        
        # 各分割パス
        train_path = os.path.join(args.data_dir, "train")
        valid_path = os.path.join(args.data_dir, "valid")
        test_path = os.path.join(args.data_dir, "test")
        
        # 訓練データからサンプリング
        if os.path.exists(train_path):
            for i in range(args.samples_per_split):
                sample = get_sample_from_dataset(processor, train_path, f"訓練 #{i+1}")
                if sample:
                    samples.append(sample)
        else:
            print(f"警告: 訓練データディレクトリが存在しません: {train_path}")
        
        # 検証データからサンプリング
        if os.path.exists(valid_path):
            for i in range(args.samples_per_split):
                sample = get_sample_from_dataset(processor, valid_path, f"検証 #{i+1}")
                if sample:
                    samples.append(sample)
        else:
            print(f"警告: 検証データディレクトリが存在しません: {valid_path}")
        
        # テストデータからサンプリング
        if os.path.exists(test_path):
            for i in range(args.samples_per_split):
                sample = get_sample_from_dataset(processor, test_path, f"テスト #{i+1}")
                if sample:
                    samples.append(sample)
        else:
            print(f"警告: テストデータディレクトリが存在しません: {test_path}")
    
    # 単一データセットパスが指定されている場合
    elif args.dataset_path:
        print(f"\n=== 単一データセットからサンプリング ===")
        sample = get_sample_from_dataset(processor, args.dataset_path)
        if sample:
            samples.append(sample)
    
    # データディレクトリだけが指定されている場合（デフォルトでテストデータを使用）
    elif args.data_dir:
        test_path = os.path.join(args.data_dir, "test")
        if os.path.exists(test_path):
            sample = get_sample_from_dataset(processor, test_path, "テスト")
            if sample:
                samples.append(sample)
        else:
            print(f"警告: テストデータディレクトリが存在しません: {test_path}")
    
    # サンプルが一つもない場合
    if not samples:
        print("エラー: サンプルテキストを取得できませんでした")
        print("テストを終了します。")
        return
    
    print(f"\n合計 {len(samples)} 件のサンプルを取得しました")
    
    # 各サンプルに対してテストを実行
    for i, sample in enumerate(samples):
        print(f"\n\n=== サンプル #{i+1}/{len(samples)} のテスト ===")
        sample_text = sample["text"]
        
        # トークン化
        if "input_ids" in sample and sample["source"] == "input_ids":
            tokens = sample["input_ids"]
            print(f"既存のトークンIDを使用: {len(tokens)} トークン")
        else:
            tokens = processor.tokenize_text(sample_text)
            print(f"トークン化完了: {len(tokens)} トークン")
        
        # 元のテキストをデコード
        decoded_text = processor.tokenizer.decode(tokens)
        print(f"元のテキスト（デコード後）: {decoded_text[:100]}..." if len(decoded_text) > 100 else decoded_text)
        
        # トークンの詳細
        print(f"最初の20トークン: {tokens[:20]}")
        
        # マスキングのテスト
        token_tensor = torch.tensor([tokens])
        noisy_tokens, noise_mask = processor.add_noise(token_tensor, args.mask_ratio)
        noisy_tokens_list = noisy_tokens[0].tolist()
        
        # マスクの数を確認
        mask_count = noisy_tokens_list.count(processor.mask_token_id)
        print(f"マスクトークン数: {mask_count}/{len(noisy_tokens_list)} ({mask_count/len(noisy_tokens_list)*100:.1f}%)")
        
        # マスクしたトークンをデコード
        masked_decoded = processor.tokenizer.decode(noisy_tokens_list)
        print(f"マスク後のテキスト: {masked_decoded[:100]}..." if len(masked_decoded) > 100 else masked_decoded)
        
        # Diffusionモデル用のバッチ準備テスト
        batch = {
            "input_ids": token_tensor,
            "attention_mask": torch.tensor([[1] * len(tokens)])
        }
        
        # Diffusionモデル用のバッチを準備
        diffusion_batch = processor.prepare_diffusion_batch(batch, args.mask_ratio)
        
        # 結果を表示
        print("\nDiffusionモデル用のバッチ:")
        for key, value in diffusion_batch.items():
            print(f"{key}の形状: {value.shape}")
        
        # マスクの割合を確認
        mask_count = (diffusion_batch["input_ids"] == processor.mask_token_id).sum().item()
        total_tokens = diffusion_batch["input_ids"].numel()
        print(f"マスクトークンの割合: {mask_count}/{total_tokens} ({mask_count/total_tokens*100:.1f}%)")
        
        # 予測対象（ラベル != -100）の数を確認
        label_count = (diffusion_batch["labels"] != -100).sum().item()
        print(f"予測対象の割合: {label_count}/{total_tokens} ({label_count/total_tokens*100:.1f}%)")
        
        # マスクとラベルが一致するか確認
        mask_positions = (diffusion_batch["input_ids"] == processor.mask_token_id)
        label_positions = (diffusion_batch["labels"] != -100)
        match_count = (mask_positions == label_positions).sum().item()
        print(f"マスク位置とラベル位置の一致率: {match_count}/{total_tokens} ({match_count/total_tokens*100:.1f}%)")
    
    print("\nデータプロセッサーのテストが完了しました")

if __name__ == "__main__":
    main()