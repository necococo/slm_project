#!/usr/bin/env python
# coding: utf-8
"""
Wiki40B-ja用トレーニングモデルのテストスクリプト
学習したモデルの検証とテキスト生成機能のテストを行います
"""

import os
import sys
import argparse
import torch
import time
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# 必要なモジュールへのパスを追加
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from slm.config import ModelConfig
from slm.modules.wave_network import WaveNetworkLM
# 標準のトークナイザーを使用するのでJapaneseTokenizerは不要
from slm.diffusion import SimpleTextDiffusion


def parse_args():
    parser = argparse.ArgumentParser(description="Wiki40B-jaモデルのテスト")
    
    # モデルパス関連
    parser.add_argument("--model_path", type=str, required=True,
                        help="テストするモデルのパス")
    parser.add_argument("--tokenizer_name", type=str, default="megagonlabs/t5-base-japanese-web",
                        help="使用するトークナイザー名")
    
    # データセット関連
    parser.add_argument("--dataset_name", type=str, default="toramaru-u/wiki40b-ja",
                        help="テスト用データセット名")
    parser.add_argument("--use_local_dataset", action="store_true",
                        help="ローカルデータセットを使用するか")
    parser.add_argument("--local_data_dir", type=str, default="./data/wiki40b_ja",
                        help="ローカルデータセットのディレクトリ")
    
    # テスト設定
    parser.add_argument("--num_samples", type=int, default=10,
                        help="テキスト生成のサンプル数")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="最大シーケンス長")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="バッチサイズ")
    parser.add_argument("--diffusion_steps", type=int, default=20,
                        help="拡散ステップ数")
    parser.add_argument("--temperature", type=float, default=1.0, 
                        help="生成時の温度パラメータ（高いほど多様性が増す）")
    parser.add_argument("--seed", type=int, default=42,
                        help="乱数シード")
    
    # テスト種類
    parser.add_argument("--run_perplexity", action="store_true",
                        help="パープレキシティテストを実行")
    parser.add_argument("--run_generation", action="store_true", 
                        help="テキスト生成テストを実行")
    parser.add_argument("--run_benchmark", action="store_true",
                        help="パフォーマンスベンチマークを実行")
    
    return parser.parse_args()


def load_tokenizer(tokenizer_name):
    """トークナイザーをロードする関数"""
    print(f"トークナイザー {tokenizer_name} をロード中...")
    
    # HuggingFaceからロード
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # マスクトークンの追加（T5トークナイザーには必要に応じて）
    if not hasattr(tokenizer, 'mask_token') or tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '<mask>'})
        print(f"マスクトークン '<mask>' を追加しました。")
    
    # BOSトークンの追加（必要に応じて）
    if not hasattr(tokenizer, 'bos_token') or tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '<s>'})
        print(f"BOSトークン '<s>' を追加しました。")
    
    print(f"トークナイザーをロードしました。語彙サイズ: {len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else tokenizer.vocab_size}")
    
    return tokenizer


def load_model(model_path, tokenizer):
    """モデルをロードする関数"""
    print(f"モデルを {model_path} からロード中...")
    
    try:
        # モデル設定ファイルの読み込み（存在する場合）
        config_path = os.path.join(model_path, "model_config.pt")
        if os.path.exists(config_path):
            model_config = torch.load(config_path)
            print("モデル設定をファイルから読み込みました")
        else:
            # 設定がない場合はデフォルト設定を使用
            model_config = ModelConfig(
                hidden_size=1024,
                num_layers=3,
                vocab_size=tokenizer.vocab_size,
                max_seq_len=512,
                dropout_prob=0.1,
                use_rope=True,
                use_wavelet=True,
                wavelet_name="haar",
                activation="gelu"
            )
            print("モデル設定ファイルが見つからないため、デフォルト設定を使用します")
        
        # トークナイザーを設定
        model_config.set_tokenizer(tokenizer)
        
        # モデルのインスタンス化
        model = WaveNetworkLM(model_config)
        
        # モデルの重みを読み込む
        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            # ファイル名が異なる場合、ディレクトリ内の.binまたは.ptファイルを探す
            for filename in os.listdir(model_path):
                if filename.endswith(".bin") or filename.endswith(".pt"):
                    if "config" not in filename:  # configファイルは除外
                        checkpoint_path = os.path.join(model_path, filename)
                        print(f"代替チェックポイントファイルを見つけました: {checkpoint_path}")
                        break
        
        # チェックポイントをロード
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # キーの前処理（必要に応じてモジュール名の前にある'model.'を削除）
        if all(k.startswith('model.') for k in checkpoint.keys() if k != 'epoch'):
            # 'model.'が接頭辞として付いているため、それを取り除く
            new_state_dict = {k[6:]: v for k, v in checkpoint.items() if k != 'epoch'}
            model.load_state_dict(new_state_dict)
        else:
            # そのまま読み込む
            model.load_state_dict(checkpoint)
        
        print(f"モデルを正常にロードしました")
        return model
    
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_test_dataset(args, tokenizer):
    """テスト用データセットを読み込む関数"""
    if args.use_local_dataset:
        print(f"ローカルデータセットを使用: {args.local_data_dir}")
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(args.local_data_dir)
            print("ローカルデータセットを正常にロードしました")
        except Exception as e:
            print(f"ローカルデータセットのロード中にエラーが発生しました: {e}")
            print(f"Hugging Faceからデータセットをロードします")
            dataset = load_dataset(args.dataset_name)
    else:
        print(f"Hugging Faceからデータセット {args.dataset_name} をロード中...")
        dataset = load_dataset(args.dataset_name)
    
    print("データセット情報:")
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])}サンプル")
    
    return dataset


def calculate_perplexity(model, dataset, tokenizer, args):
    """パープレキシティを計算する関数"""
    print("\n===== パープレキシティ評価 =====")
    model.eval()
    
    # GPUが利用可能であればそれを使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    total_loss = 0.0
    total_tokens = 0
    
    # どのスプリットを使うか判断する
    available_splits = list(dataset.keys())
    print(f"利用可能なスプリット: {available_splits}")
    
    # テストが存在すればテスト、なければバリデーション、それもなければトレインを使用
    test_split = "test" if "test" in dataset else "validation" if "validation" in dataset else "train"
    
    # サンプル数を決定（最大100または全体）
    num_samples = min(args.num_samples if args.num_samples > 0 else 100, len(dataset[test_split]))
    
    # テスト用データのバッチサイズ
    batch_size = args.batch_size
    
    print(f"{test_split}セットを使用して評価します。サンプル数: {num_samples}")
    
    # プログレスバー
    progress_bar = tqdm(range(0, num_samples, batch_size), desc="パープレキシティ評価")
    
    with torch.no_grad():
        for start_idx in progress_bar:
            end_idx = min(start_idx + batch_size, num_samples)
            batch_texts = [dataset[test_split][i]["text"] for i in range(start_idx, end_idx)]
            
            # テキストをトークン化
            batch_input_ids = []
            batch_attention_mask = []
            
            for text in batch_texts:
                # トークン化 - トークナイザーを使用
                token_ids = tokenizer._tokenizer.encode(text, add_special_tokens=False)
                
                # 最大長で切り詰め
                if len(token_ids) > args.max_seq_len:
                    token_ids = token_ids[:args.max_seq_len]
                
                # 注意マスクを作成
                attn_mask = [1] * len(token_ids)
                
                batch_input_ids.append(token_ids)
                batch_attention_mask.append(attn_mask)
            
            # パディング処理
            max_len = max([len(ids) for ids in batch_input_ids])
            padded_input_ids = []
            padded_attention_mask = []
            
            for input_ids, attention_mask in zip(batch_input_ids, batch_attention_mask):
                padding_length = max_len - len(input_ids)
                padded_input_ids.append(input_ids + [tokenizer.pad_token_id] * padding_length)
                padded_attention_mask.append(attention_mask + [0] * padding_length)
            
            # テンソルに変換
            input_ids_tensor = torch.tensor(padded_input_ids).to(device)
            attention_mask_tensor = torch.tensor(padded_attention_mask).to(device)
            
            # モデルの出力を取得（損失を計算）
            outputs = model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                labels=input_ids_tensor  # 入力と同じものをラベルとして使用
            )
            
            loss = outputs.loss.item()
            total_loss += loss * (end_idx - start_idx)  # バッチサイズを掛ける
            
            # トークン数を累積（パディングを除く）
            total_tokens += torch.sum(attention_mask_tensor).item()
            
            # プログレスバーを更新
            progress_bar.set_postfix(loss=f"{loss:.4f}")
    
    # パープレキシティを計算
    avg_loss = total_loss / num_samples
    perplexity = np.exp(avg_loss)
    
    print(f"\n評価結果:")
    print(f"  サンプル数: {num_samples}")
    print(f"  平均損失: {avg_loss:.4f}")
    print(f"  パープレキシティ (PPL): {perplexity:.4f}")
    
    return perplexity


def generate_text_with_diffusion(model, tokenizer, args):
    """
    拡散モデルを使ってテキストを生成する関数
    """
    print("\n===== テキスト生成テスト =====")
    model.eval()

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 拡散モデルを初期化
    mask_token_id = tokenizer.mask_token_id
    vocab_size = model.get_classifier_weights().size(0)
    diffuser = SimpleTextDiffusion(
        timesteps=args.diffusion_steps,
        mask_token_id=mask_token_id,
        vocab_size=vocab_size
    )
    diffuser = diffuser.to(device)

    # シード設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # テキスト生成
    max_length = args.max_seq_len
    n_samples = args.num_samples
    temperature = args.temperature

    print(f"パラメータ設定: タイムステップ={args.diffusion_steps}, 温度={temperature}, 生成数={n_samples}")

    for sample_idx in range(n_samples):
        # 時間計測開始
        start_time = time.time()
        
        # ランダムなノイズから始める (完全にランダムなトークンIDを生成)
        # 制御可能にするためにマスクから始めることもできる
        sample_length = np.random.randint(64, max_length)  # ランダムな長さ
        
        # ランダムまたはマスクトークンから開始
        use_mask_start = True  # マスクトークンから開始する場合はTrue
        
        if use_mask_start:
            # すべてマスクトークンから始める
            input_ids = torch.full((1, sample_length), mask_token_id, device=device)
        else:
            # ランダムなトークンからスタート
            input_ids = torch.randint(
                0, tokenizer._tokenizer.vocab_size, (1, sample_length), device=device
            )
        
        # 注意マスク (すべて1)
        attention_mask = torch.ones_like(input_ids)
        
        # 拡散サンプリングを繰り返して最終的なテキストを生成
        for t in tqdm(range(args.diffusion_steps - 1, -1, -1), desc=f"サンプル {sample_idx+1}/{n_samples}"):
            # スケジュール用のタイムステップ
            timestep = torch.full((1,), t, device=device, dtype=torch.long)
            
            # 条件付き拡散サンプリングステップ
            with torch.no_grad():
                # モデルで次のステップを予測
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).logits
                
                # 温度でロジットをスケーリング
                if temperature != 1.0:
                    logits = logits / temperature
                
                # 現在のタイムステップでトークンをサンプリング
                input_ids = diffuser.p_sample(
                    input_ids, logits, timestep
                )
        
        # 生成されたテキストをデコード
        generated_text = tokenizer.decode(input_ids[0].cpu().tolist())
        
        generation_time = time.time() - start_time
        
        print(f"\nサンプル {sample_idx+1} (長さ: {sample_length}, 生成時間: {generation_time:.2f}秒):")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
        
    return True


def run_benchmark(model, tokenizer, args):
    """
    モデルのパフォーマンスをベンチマークする関数
    """
    print("\n===== パフォーマンスベンチマーク =====")
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"実行デバイス: {device}")
    
    model.eval()
    model = model.to(device)
    
    # さまざまなバッチサイズとシーケンス長でベンチマーク
    batch_sizes = [1, 4, 8] if torch.cuda.is_available() else [1, 2, 4]
    seq_lengths = [128, 256, 512]
    
    print(f"{'バッチサイズ':<12}{'シーケンス長':<14}{'推論時間 (ms)':<16}{'トークン/秒'}")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            # ランダム入力を生成
            input_ids = torch.randint(
                0, tokenizer._tokenizer.vocab_size, 
                (batch_size, seq_len), 
                device=device
            )
            attention_mask = torch.ones_like(input_ids)
            
            # ウォームアップ実行
            with torch.no_grad():
                for _ in range(3):
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 本番実行と計測
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            repeat_count = 10
            with torch.no_grad():
                for _ in range(repeat_count):
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            # 平均時間を計算
            avg_time_ms = (end_time - start_time) * 1000 / repeat_count
            
            # トークン/秒を計算
            tokens_per_second = (batch_size * seq_len) / (avg_time_ms / 1000)
            
            print(f"{batch_size:<12}{seq_len:<14}{avg_time_ms:.2f} ms{tokens_per_second:.2f}")
    
    return True


def main():
    """メイン関数"""
    args = parse_args()
    
    # 乱数シードの設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # トークナイザーのロード
    tokenizer = load_tokenizer(args.tokenizer_name)
    
    # モデルのロード
    model = load_model(args.model_path, tokenizer)
    if model is None:
        print("モデルのロードに失敗しました。終了します。")
        return
    
    # モデル情報の表示
    param_count = sum(p.numel() for p in model.parameters())
    print(f"モデル情報: {param_count:,}パラメータ")
    
    # パープレキシティ評価
    if args.run_perplexity:
        # テストデータセットのロード
        dataset = load_test_dataset(args, tokenizer)
        perplexity = calculate_perplexity(model, dataset, tokenizer, args)
    
    # テキスト生成テスト
    if args.run_generation:
        generate_text_with_diffusion(model, tokenizer, args)
    
    # パフォーマンスベンチマーク
    if args.run_benchmark:
        run_benchmark(model, tokenizer, args)
    
    print("\nテスト完了")


if __name__ == "__main__":
    main()