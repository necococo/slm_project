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
    parser.add_argument("--test_text", type=str, 
                        default="フッカーがリー軍に自軍を攻撃させようとした戦術は明らかに概念として健全だが、"
                                "フッカーとその部下達が行った方法には恐ろしく欠陥があった。"
                                "実際の戦闘では北軍がリーのそれまで「無敵の」兵士達と同じくらい戦闘...",
                        help="テスト用テキスト")
    parser.add_argument("--timesteps", type=int, default=10,
                        help="タイムステップ数")
    parser.add_argument("--mask_prob_min", type=float, default=0.0,
                        help="最小マスク確率")
    parser.add_argument("--mask_prob_max", type=float, default=0.9,
                        help="最大マスク確率")
    
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
    
    # テキストのトークン化
    tokens = processor.tokenize_text(args.test_text)
    token_tensor = torch.tensor([tokens])
    
    print(f"\nオリジナルテキスト:")
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
        custom_decoded = processor.decode(noisy_tokens_list)
        
        print(f"\n=== タイムステップ {t}/{args.timesteps-1} ===")
        print(f"マスク率: {mask_ratio:.2f} (期待値: {diffusion.mask_probs[t].item():.2f})")
        print(f"マスクトークン数: {noisy_tokens_list.count(processor.mask_token_id)}/{len(noisy_tokens_list)}")
        
        print("\n標準デコード結果:")
        print(standard_decoded[:200] + "..." if len(standard_decoded) > 200 else standard_decoded)
        
        print("\n専用デコード結果:")
        print(custom_decoded[:200] + "..." if len(custom_decoded) > 200 else custom_decoded)
    
    print("\nシンプルなDiffusionモデルのテストが完了しました")

if __name__ == "__main__":
    main()