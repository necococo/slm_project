#!/usr/bin/env python
# coding: utf-8
# tests/test_simple_diffusion.py
# シンプルなDiffusionモデルのテスト用スクリプト

import os
import sys
import argparse
import torch
# slmモジュールからインポートするよう修正
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slm.data_processor import SimpleDataProcessor
from slm.simple_diffusion import SimpleMaskingDiffusion

def parse_args():
    parser = argparse.ArgumentParser(description="シンプルなDiffusionモデルのテスト")
    
    parser.add_argument("--tokenizer_name", type=str, default="megagonlabs/t5-base-japanese-web",
                        help="使用するトークナイザー名")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大シーケンス長")
    parser.add_argument("--mask_token", type=str, default="<mask>",
                        help="マスクトークン")
    parser.add_argument("--timesteps", type=int, default=5,
                        help="タイムステップ数")
    parser.add_argument("--mask_prob_min", type=float, default=0.0,
                        help="最小マスク確率")
    parser.add_argument("--mask_prob_max", type=float, default=0.8,
                        help="最大マスク確率")
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
        
        # サンプルからテキストを抽出
        if "text" in random_sample:
            sample_text = random_sample["text"]
            print(f"\n=== {split_name}データセットから選択したサンプル（インデックス: {random_idx}/{dataset_size-1}） ===")
            print(f"テキスト（プレビュー）: {sample_text[:100]}..." if len(sample_text) > 100 else sample_text)
            return sample_text
        elif "input_ids" in random_sample:
            # トークンIDからテキストを復元
            input_ids = random_sample["input_ids"]
            sample_text = processor.tokenizer.decode(input_ids)
            print(f"\n=== {split_name}データセットから選択したサンプル（インデックス: {random_idx}/{dataset_size-1}） ===")
            print(f"デコードテキスト（プレビュー）: {sample_text[:100]}..." if len(sample_text) > 100 else sample_text)
            return sample_text
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
    
    # サンプルテキストのリスト
    sample_texts = []
    
    # 引数でテキストが直接指定されている場合
    if args.test_text:
        sample_texts.append(args.test_text)
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
                    sample_texts.append(sample)
        else:
            print(f"警告: 訓練データディレクトリが存在しません: {train_path}")
        
        # 検証データからサンプリング
        if os.path.exists(valid_path):
            for i in range(args.samples_per_split):
                sample = get_sample_from_dataset(processor, valid_path, f"検証 #{i+1}")
                if sample:
                    sample_texts.append(sample)
        else:
            print(f"警告: 検証データディレクトリが存在しません: {valid_path}")
        
        # テストデータからサンプリング
        if os.path.exists(test_path):
            for i in range(args.samples_per_split):
                sample = get_sample_from_dataset(processor, test_path, f"テスト #{i+1}")
                if sample:
                    sample_texts.append(sample)
        else:
            print(f"警告: テストデータディレクトリが存在しません: {test_path}")
    
    # 単一データセットパスが指定されている場合
    elif args.dataset_path:
        print(f"\n=== 単一データセットからサンプリング ===")
        sample = get_sample_from_dataset(processor, args.dataset_path)
        if sample:
            sample_texts.append(sample)
    
    # データディレクトリだけが指定されている場合（デフォルトでテストデータを使用）
    elif args.data_dir:
        test_path = os.path.join(args.data_dir, "test")
        if os.path.exists(test_path):
            sample = get_sample_from_dataset(processor, test_path, "テスト")
            if sample:
                sample_texts.append(sample)
        else:
            print(f"警告: テストデータディレクトリが存在しません: {test_path}")
    
    # サンプルが一つもない場合
    if not sample_texts:
        print("エラー: サンプルテキストを取得できませんでした")
        print("テストを終了します。")
        return
    
    print(f"\n合計 {len(sample_texts)} 件のサンプルテキストを取得しました")
    
    # 各サンプルに対してDiffusionモデルをテスト
    for i, sample_text in enumerate(sample_texts):
        print(f"\n\n=== サンプル #{i+1}/{len(sample_texts)} の処理 ===")
        
        # テキストのトークン化
        tokens = processor.tokenize_text(sample_text)
        token_tensor = torch.tensor([tokens])
        
        # テキストの表示
        print(f"オリジナルテキスト:")
        print(processor.tokenizer.decode(tokens))
        
        # SimpleMaskingDiffusionの初期化
        print("\nSimpleMaskingDiffusionを初期化中...")
        diffusion = SimpleMaskingDiffusion(
            mask_token_id=processor.mask_token_id,
            mask_prob_min=args.mask_prob_min,
            mask_prob_max=args.mask_prob_max,
            timesteps=args.timesteps
        )
        
        # 各タイムステップでのノイズ追加をテスト
        print("\n各タイムステップでのノイズテスト:")
        for t in range(args.timesteps):
            # タイムステップtでノイズを追加
            t_tensor = torch.tensor([t])
            result = diffusion(token_tensor, t_tensor)
            
            noisy_tokens = result["input_ids"]
            mask = result["mask"]
            
            # マスク率を計算
            mask_ratio = mask.float().mean().item()
            
            # デコード
            noisy_tokens_list = noisy_tokens[0].tolist()
            
            # テキストデコード
            standard_decoded = processor.tokenizer.decode(noisy_tokens_list)
            
            print(f"\n=== タイムステップ {t}/{args.timesteps-1} ===")
            print(f"マスク率: {mask_ratio:.2f} (期待値: {diffusion.mask_probs[t].item():.2f})")
            print(f"マスクトークン数: {noisy_tokens_list.count(processor.mask_token_id)}/{len(noisy_tokens_list)}")
            
            print("\n標準デコード結果:")
            print(standard_decoded[:200] + "..." if len(standard_decoded) > 200 else standard_decoded)
    
    print("\nすべてのサンプルに対するDiffusionモデルのテストが完了しました")

if __name__ == "__main__":
    main()