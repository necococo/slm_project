#!/usr/bin/env python
# coding: utf-8
# tests/run_tests.py
# テスト実行ユーティリティ

import os
import sys
import subprocess
import argparse

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description="SLMプロジェクトのテスト実行")
    
    parser.add_argument("--test", type=str, default="all",
                        choices=["data", "diffusion", "all"],
                        help="実行するテスト (data: データ処理, diffusion: 拡散モデル, all: すべて)")
    parser.add_argument("--tokenizer", type=str, default="megagonlabs/t5-base-japanese-web",
                        help="使用するトークナイザー")
    parser.add_argument("--data_dir", type=str, default="/content/drive/MyDrive/slm/data/fujiki",
                        help="前処理済みデータセットのディレクトリ")
    
    return parser.parse_args()

def run_data_processor_test(args):
    """データ処理のテストを実行"""
    print("=== データプロセッサーのテスト実行 ===")
    
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "test_data_processor.py"),
        f"--tokenizer_name={args.tokenizer}",
        f"--data_dir={args.data_dir}",
        "--sample_all_splits"
    ]
    subprocess.run(cmd)

def run_diffusion_test(args):
    """拡散モデルのテストを実行"""
    print("=== 拡散モデルのテスト実行 ===")
    
    # 各データセットのパス
    train_data_path = os.path.join(args.data_dir, "train")
    valid_data_path = os.path.join(args.data_dir, "valid")
    test_data_path = os.path.join(args.data_dir, "test")
    
    # 全データフォルダを親ディレクトリとして渡す
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "test_simple_diffusion.py"),
        f"--tokenizer_name={args.tokenizer}",
        f"--data_dir={args.data_dir}",
        "--timesteps=5",
        "--mask_prob_min=0.0",
        "--mask_prob_max=0.8",
        "--sample_all_splits"
    ]
    subprocess.run(cmd)

def main():
    args = parse_args()
    
    if args.test in ["data", "all"]:
        run_data_processor_test(args)
        print("\n")
    
    if args.test in ["diffusion", "all"]:
        run_diffusion_test(args)

if __name__ == "__main__":
    main()