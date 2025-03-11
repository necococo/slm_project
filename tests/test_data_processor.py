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
    # parser.add_argument("--test_text", type=str, 
    #                     default="フッカーがリー軍に自軍を攻撃させようとした戦術は明らかに概念として健全だが、"
    #                             "フッカーとその部下達が行った方法には恐ろしく欠陥があった。"
    #                             "実際の戦闘では北軍がリーのそれまで「無敵の」兵士達と同じくらい戦闘...",
    #                     help="テスト用テキスト")
    parser.add_argument("--mask_ratio", type=float, default=0.2,
                        help="マスク割合")
    parser.add_argument("--dataset_path", type=str, default="/content/drive/MyDrive/slm/data/wiki40b_ja/train",
                        help="データセットのパス（指定するとデータセットのテストも行う）")
    
    return parser.parse_args()

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
    print(f"テストテキスト: {args.test_text[:50]}...")
    print(f"マスク割合: {args.mask_ratio:.2f}")
    
    # データセットのテスト（指定されている場合）
    if args.dataset_path:
        # データセットのロード
        dataset = processor.load_dataset(args.dataset_path)
        dataset_size = len(dataset)
        
        # ランダムにサンプルを選択
        import random
        random_idx = random.randint(0, dataset_size - 1)
        random_sample = dataset[random_idx]
        
        # サンプルのテキストを取得
        if "text" in random_sample:
            sample_text = random_sample["text"]
            print(f"\n=== データセットからランダムに選択したサンプル（インデックス: {random_idx}/{dataset_size-1}） ===")
            print(f"元のテキスト（プレビュー）: {sample_text[:100]}..." if len(sample_text) > 100 else sample_text)
            
            # マスキングとデコードのテスト
            processor.test_decode_with_mask(sample_text, args.mask_ratio)
        else:
            print("警告: 選択したサンプルに'text'フィールドがありません")
            print(f"サンプルのキー: {list(random_sample.keys())}")
            # フォールバックとして引数のテキストを使用
            processor.test_decode_with_mask(args.test_text, args.mask_ratio)
    else:
        # データセットパスが指定されていない場合は引数のテキストを使用
        processor.test_decode_with_mask(args.test_text, args.mask_ratio)
    
    # 追加のデータセットテスト（指定されている場合）
    if args.dataset_path and not "text" in random_sample:
        print("\n=== データセットのテスト ===")
        try:
            # データセットのロード
            dataset = processor.load_dataset(args.dataset_path)
            print(f"データセットサイズ: {len(dataset)}")
            
            # 最初のサンプルを表示
            if len(dataset) > 0:
                first_sample = dataset[0]
                print("\n最初のサンプル:")
                
                if "text" in first_sample:
                    # テキストデータ
                    text = first_sample["text"]
                    print(f"テキスト: {text[:100]}..." if len(text) > 100 else f"テキスト: {text}")
                    
                    # トークン化
                    tokens = processor.tokenize_text(text)
                    print(f"トークン数: {len(tokens)}")
                    print(f"最初の20トークン: {tokens[:20]}")
                    
                    # デコード
                    decoded_text = processor.tokenizer.decode(tokens)
                    print(f"デコードされたテキスト: {decoded_text[:100]}..." if len(decoded_text) > 100 else decoded_text)
                    
                    # マスキングのテスト
                    token_tensor = torch.tensor([tokens])
                    noisy_tokens, noise_mask = processor.add_noise(token_tensor, args.mask_ratio)
                    noisy_tokens_list = noisy_tokens[0].tolist()
                    
                    # マスクの数を確認
                    mask_count = noisy_tokens_list.count(processor.mask_token_id)
                    print(f"マスクトークン数: {mask_count}/{len(noisy_tokens_list)} ({mask_count/len(noisy_tokens_list)*100:.1f}%)")
                    
                    # デコード
                    standard_decoded = processor.tokenizer.decode(noisy_tokens_list)
                    
                    print(f"標準デコード結果: {standard_decoded[:100]}..." if len(standard_decoded) > 100 else standard_decoded)
                    
                elif "input_ids" in first_sample:
                    # 既にトークン化済み
                    input_ids = first_sample["input_ids"]
                    print(f"トークン数: {len(input_ids)}")
                    print(f"最初の20トークン: {input_ids[:20]}")
                    
                    # デコード
                    decoded_text = processor.tokenizer.decode(input_ids)
                    print(f"デコードされたテキスト: {decoded_text[:100]}..." if len(decoded_text) > 100 else decoded_text)
                
                # Diffusionモデル用のバッチ準備テスト
                if "input_ids" in first_sample:
                    # バッチの形式に変換
                    batch = {
                        "input_ids": torch.tensor([first_sample["input_ids"]]),
                        "attention_mask": torch.tensor([first_sample.get("attention_mask", [1] * len(first_sample["input_ids"]))])
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
            
        except Exception as e:
            print(f"データセットのテスト中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nデータプロセッサーのテストが完了しました")

if __name__ == "__main__":
    main()