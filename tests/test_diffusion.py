#!/usr/bin/env python
# coding: utf-8
# tests/test_diffusion.py
# 実際のDiffusionモデル実装のテスト用スクリプト

import os
import sys
import argparse
import torch
# slmモジュールからインポートするよう修正
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slm.data_processor import SimpleDataProcessor
from slm.diffusion import SimpleTextDiffusion

def parse_args():
    parser = argparse.ArgumentParser(description="本番用Diffusionモデルのテスト")
    
    parser.add_argument("--tokenizer_name", type=str, default="megagonlabs/t5-base-japanese-web",
                        help="使用するトークナイザー名")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大シーケンス長")
    parser.add_argument("--mask_token", type=str, default="<mask>",
                        help="マスクトークン")
    parser.add_argument("--timesteps", type=int, default=5,
                        help="タイムステップ数")
    parser.add_argument("--beta_schedule", type=str, default="linear",
                        help="ノイズスケジュール (linear, cosine)")
    parser.add_argument("--data_dir", type=str, default="/content/drive/MyDrive/slm/data/fujiki/wiki40b_ja",
                        help="前処理済みデータセットのディレクトリ")
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
    
    # データディレクトリのみ指定の場合（デフォルトでテストデータを使用）
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
        print("代わりにデフォルトのテストテキストを使用します")
        sample_texts.append({
            "text": "これはDiffusionモデルのテストです。日本語でテキスト生成の実験を行っています。",
            "source": "default"
        })
    
    print(f"\n合計 {len(sample_texts)} 件のサンプルテキストを取得しました")
    
    # 各サンプルに対してDiffusionモデルをテスト
    for i, sample in enumerate(sample_texts):
        sample_text = sample["text"]
        print(f"\n\n=== サンプル #{i+1}/{len(sample_texts)} の処理 ===")
        
        # テキストのトークン化
        if "input_ids" in sample and sample["source"] == "input_ids":
            tokens = sample["input_ids"]
            print(f"既存のトークンIDを使用: {len(tokens)} トークン")
        else:
            tokens = processor.tokenize_text(sample_text)
            print(f"トークン化完了: {len(tokens)} トークン")
        
        token_tensor = torch.tensor([tokens])
        
        # オリジナルテキストを表示
        print(f"オリジナルテキスト:")
        print(processor.tokenizer.decode(tokens))
        
        # SimpleTextDiffusionの初期化
        print("\nSimpleTextDiffusionを初期化中...")
        diffusion = SimpleTextDiffusion(
            timesteps=args.timesteps,
            mask_token_id=processor.mask_token_id,
            vocab_size=processor.tokenizer.vocab_size,
            beta_schedule=args.beta_schedule
        )
        
        # 各タイムステップでのノイズ追加をテスト
        print("\n各タイムステップでのノイズテスト:")
        for t in range(args.timesteps):
            # タイムステップtでノイズを追加
            t_tensor = torch.tensor([t])
            
            # SimpleTextDiffusion.forwardを呼び出し
            x_noised, noise_mask = diffusion.add_noise(token_tensor, t_tensor)
            
            # マスク率を計算
            mask_ratio = noise_mask.float().mean().item()
            
            # デコード
            noisy_tokens_list = x_noised[0].tolist()
            
            # テキストデコード
            standard_decoded = processor.tokenizer.decode(noisy_tokens_list)
            
            print(f"\n=== タイムステップ {t}/{args.timesteps-1} ===")
            print(f"マスク率: {mask_ratio:.2f}")
            print(f"マスクトークン数: {noisy_tokens_list.count(processor.mask_token_id)}/{len(noisy_tokens_list)}")
            
            print("\n標準デコード結果:")
            print(standard_decoded[:200] + "..." if len(standard_decoded) > 200 else standard_decoded)
        
        # リバースプロセスのテスト（もし実装されていれば）
        if hasattr(diffusion, "denoise") and callable(diffusion.denoise):
            print("\n=== リバースプロセス（ノイズ除去）のテスト ===")
            # 完全にマスクしたテキスト（t=T-1）
            t_tensor = torch.tensor([args.timesteps - 1])
            x_noised, _ = diffusion.add_noise(token_tensor, t_tensor)
            
            # デコード前の状態を表示
            noisy_tokens_list = x_noised[0].tolist()
            print(f"マスク適用後（タイムステップ {args.timesteps-1}）:")
            print(processor.tokenizer.decode(noisy_tokens_list))
            
            # リバースプロセス
            print("\nリバースプロセス（ノイズ除去）を実行中...")
            denoised_tokens = diffusion.denoise(x_noised, t_tensor)
            
            # 結果を表示
            denoised_tokens_list = denoised_tokens[0].tolist()
            print(f"\nノイズ除去後:")
            print(processor.tokenizer.decode(denoised_tokens_list))
    
    print("\n本番Diffusionモデルのテストが完了しました")

if __name__ == "__main__":
    main()