#!/usr/bin/env python
# coding: utf-8
# slm/run_eval.py - 保存されたモデルの評価を実行するスクリプト
# Why: チェックポイントから重みを読み込み、複数の評価指標で言語モデルの性能を測定する

import os
import argparse
import torch
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from typing import Dict, Any, List, Optional

from slm.config import ModelConfig
from slm.modules.wave_network import WaveNetworkLM
from slm.evaluation import (
    evaluate_perplexity, 
    evaluate_bleu, 
    evaluate_rouge,
    temperature_sampling_decode
)
from slm.tokenizer import load_tokenizer


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description="保存済みのSLMモデルの評価を実行します")
    
    # モデル・データ関連
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="評価するモデルチェックポイントのパス (例: ./outputs/checkpoints/final_model.pt)")
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="評価用データセットのパス (HuggingFaceのデータセット名またはディレクトリパス)")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="トークナイザーのパス (指定しない場合、チェックポイントから読み込み)")
    parser.add_argument("--tokenizer_name", type=str, default="megagonlabs/t5-base-japanese-web",
                        help="トークナイザー名 (tokenizer_pathが指定されていない場合に使用)")
    
    # 評価設定
    parser.add_argument("--batch_size", type=int, default=8,
                        help="評価時のバッチサイズ")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="生成するトークンの最大数")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="サンプリング温度 (1.0で等確率、小さいほどgreedy)")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="評価用サンプル数 (0の場合はデータセット全体を使用)")
    
    # 出力関連
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="評価結果の出力ディレクトリ")
    parser.add_argument("--generate_samples", type=int, default=5,
                        help="生成サンプル数 (0の場合は生成しない)")
    parser.add_argument("--device", type=str, default=None,
                        help="使用するデバイス (例: 'cuda', 'cpu', 'mps')")

    args = parser.parse_args()
    
    # デバイス自動検出
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    
    return args


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda") -> tuple[WaveNetworkLM, Any]:
    """
    How:
        チェックポイントからモデルとトークナイザーを読み込む
        
    Args:
        checkpoint_path: モデルチェックポイントのパス
        device: モデルを配置するデバイス
        
    Returns:
        モデルとトークナイザーのタプル
    """
    print(f"チェックポイント {checkpoint_path} からモデルを読み込み中...")
    
    try:
        # PyTorch 2.6以上の場合はweights_only=Falseを使用
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        # 古いPyTorchバージョンではweights_onlyがない
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
    # モデル設定を取得
    if "model_config" not in checkpoint:
        raise ValueError(f"チェックポイントにmodel_configが見つかりません: {checkpoint_path}")
        
    model_config = checkpoint["model_config"]
    print(f"モデル設定: hidden_size={model_config.hidden_size}, num_layers={model_config.num_layers}, vocab_size={model_config.vocab_size}")
    
    # モデルの初期化
    model = WaveNetworkLM(model_config)
    
    # 重みの読み込み
    try:
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("モデルの重みを正常に読み込みました")
        else:
            print("警告: チェックポイントにmodel_state_dictが見つかりません")
    except Exception as e:
        print(f"モデルの重み読み込みエラー: {e}")
        
    # モデルをデバイスに移動
    model = model.to(device)
    model.eval()  # 評価モード
    
    # トークナイザーの取得
    tokenizer = checkpoint.get("tokenizer")
    if tokenizer is None and hasattr(model_config, "tokenizer"):
        tokenizer = model_config.tokenizer
        
    return model, tokenizer


def prepare_test_dataset(args):
    """
    How:
        評価用データセットを準備する
    """
    if args.test_data_path is None:
        # デフォルトでwikitext-103の最初の数サンプルを使用
        print("テストデータセットが指定されていないため、wikitext-103からサンプルをロードします")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        if args.n_samples > 0:
            dataset = dataset.select(range(min(args.n_samples, len(dataset))))
        return dataset
    
    # ローカルディレクトリからロード
    if os.path.isdir(args.test_data_path):
        print(f"ディレクトリ {args.test_data_path} からテストデータセットをロード中...")
        try:
            dataset = load_from_disk(args.test_data_path)
            if isinstance(dataset, dict) and "test" in dataset:
                dataset = dataset["test"]
            print(f"データセットをロードしました: {len(dataset)} サンプル")
        except Exception as e:
            print(f"ローカルデータセットのロードエラー: {e}")
            return None
    # Hugging Faceからロード
    else:
        print(f"Hugging Face から {args.test_data_path} をロード中...")
        try:
            dataset = load_dataset(args.test_data_path, split="test")
            print(f"データセットをロードしました: {len(dataset)} サンプル")
        except Exception as e:
            print(f"Hugging Face データセットのロードエラー: {e}")
            return None
    
    # サンプル数の制限
    if args.n_samples > 0 and len(dataset) > args.n_samples:
        dataset = dataset.select(range(args.n_samples))
        print(f"データセットを {args.n_samples} サンプルに制限しました")
    
    return dataset


def prepare_tokenizer(args, model_tokenizer=None):
    """
    How:
        トークナイザーを準備する
    """
    if model_tokenizer is not None:
        print("モデルからトークナイザーを取得しました")
        return model_tokenizer
        
    if args.tokenizer_path:
        print(f"トークナイザーを {args.tokenizer_path} からロード中...")
        try:
            tokenizer = load_tokenizer(
                model_name=args.tokenizer_name, 
                tokenizer_path=args.tokenizer_path
            )
        except Exception as e:
            print(f"トークナイザー読み込みエラー: {e}")
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        print(f"トークナイザー {args.tokenizer_name} をダウンロード中...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        
    return tokenizer


class TestDatasetWrapper(torch.utils.data.Dataset):
    """
    How:
        テストデータセットを評価用に変換するラッパークラス
    """
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # データフォーマットの確認
        if len(dataset) > 0:
            first_item = dataset[0]
            self.keys = list(first_item.keys()) if isinstance(first_item, dict) else []
            print(f"データセットキー: {self.keys}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # データ形式に応じて変換
        if isinstance(item, dict):
            # 入力テキストを取得 (使用優先順位)
            text = None
            for key in ["text", "content", "article", "input_text"]:
                if key in item:
                    text = item[key]
                    break
                    
            if text is None and len(item) > 0:
                # 最初のキーをテキストとして使用
                text = item[list(item.keys())[0]]
            
            # トークン化
            if self.tokenizer:
                input_ids = self.tokenizer.encode(
                    text, 
                    add_special_tokens=False, 
                    max_length=self.max_length,
                    truncation=True
                )
                
                # 目標テキスト (可能であれば)
                target_text = item.get("target_text", item.get("summary", text))
                
                # 返り値は評価関数が期待する形式に
                return {
                    "input_ids": torch.tensor(input_ids),
                    "labels": torch.tensor(input_ids),  # 自己回帰評価用
                    "text": text,  # 生成サンプル用
                    "target_text": target_text  # BLEU/ROUGE評価用
                }
        
        # 文字列の場合
        elif isinstance(item, str):
            input_ids = self.tokenizer.encode(
                item, 
                add_special_tokens=False, 
                max_length=self.max_length,
                truncation=True
            )
            return {
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(input_ids),
                "text": item,
                "target_text": item
            }
        
        # その他の形式
        return {
            "input_ids": torch.tensor([]),
            "labels": torch.tensor([]),
            "text": "",
            "target_text": ""
        }


def generate_samples(model, dataset, tokenizer, device, max_new_tokens=32, temperature=1.0, num_samples=5):
    """
    How:
        モデルを使用してテキスト生成サンプルを作成
    """
    results = []
    
    # 最大num_samplesまでのサンプルを生成
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        input_ids = sample["input_ids"].to(device).unsqueeze(0)
        
        # 入力テキスト (トークナイザーで戻す)
        if tokenizer and hasattr(tokenizer, 'decode'):
            prefix = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
            # 長すぎる場合は省略
            if len(prefix) > 100:
                prefix = prefix[:100] + "..."
        else:
            # トークナイザーがない場合はテキストをそのまま使用
            prefix = sample["text"][:100] + "..." if len(sample["text"]) > 100 else sample["text"]
        
        # テキスト生成（トークナイザーを渡す）
        with torch.no_grad():
            generated_text = temperature_sampling_decode(
                model, input_ids, max_new_tokens, device, 
                temperature=temperature,
                tokenizer=tokenizer  # トークナイザーを渡す
            )
        
        results.append({
            "prefix": prefix,
            "generated": generated_text
        })
    
    return results


def main():
    """メイン実行関数"""
    # 引数の解析
    args = parse_args()
    
    # デバイスのセットアップ
    device = torch.device(args.device)
    print(f"使用デバイス: {device}")
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # モデルのロード
    model, model_tokenizer = load_model_from_checkpoint(args.checkpoint_path, str(device))
    
    # トークナイザーの準備
    tokenizer = prepare_tokenizer(args, model_tokenizer)
    
    # データセットの準備
    test_dataset = prepare_test_dataset(args)
    if test_dataset is None:
        print("テストデータセットのロードに失敗しました。終了します。")
        return
    
    # 評価用にデータセットをラップ
    wrapped_dataset = TestDatasetWrapper(test_dataset, tokenizer, max_length=512)
    print(f"評価用データセットを準備しました: {len(wrapped_dataset)} サンプル")
    
    # パープレキシティ評価
    try:
        ppl = evaluate_perplexity(model, wrapped_dataset, device, batch_size=args.batch_size)
        print(f"Perplexity: {ppl:.4f}")
        
        # 結果の保存
        with open(os.path.join(args.output_dir, "perplexity.txt"), "w") as f:
            f.write(f"Perplexity: {ppl:.4f}\n")
    except Exception as e:
        print(f"Perplexity評価でエラーが発生しました: {e}")
    
    # BLEUスコア評価
    try:
        bleu = evaluate_bleu(
            model, wrapped_dataset, device, 
            batch_size=args.batch_size, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        print(f"BLEU: {bleu:.4f}")
        
        # 結果の保存
        with open(os.path.join(args.output_dir, "bleu.txt"), "w") as f:
            f.write(f"BLEU: {bleu:.4f}\n")
    except Exception as e:
        print(f"BLEU評価でエラーが発生しました: {e}")
    
    # ROUGEスコア評価
    try:
        rouge_results = evaluate_rouge(
            model, wrapped_dataset, device, 
            batch_size=args.batch_size, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        print(f"ROUGE-1: {rouge_results['rouge1']:.4f}")
        print(f"ROUGE-2: {rouge_results['rouge2']:.4f}")
        print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
        
        # 結果の保存
        with open(os.path.join(args.output_dir, "rouge.txt"), "w") as f:
            for k, v in rouge_results.items():
                f.write(f"{k}: {v:.4f}\n")
    except Exception as e:
        print(f"ROUGE評価でエラーが発生しました: {e}")
    
    # テキスト生成サンプル
    if args.generate_samples > 0:
        try:
            samples = generate_samples(
                model, wrapped_dataset, tokenizer, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                num_samples=args.generate_samples
            )
            
            # サンプルの表示と保存
            print("\n=== 生成サンプル ===")
            with open(os.path.join(args.output_dir, "samples.txt"), "w", encoding="utf-8") as f:
                for i, sample in enumerate(samples):
                    print(f"\nサンプル {i+1}:")
                    print(f"入力: {sample['prefix']}")
                    print(f"生成: {sample['generated']}")
                    
                    f.write(f"サンプル {i+1}:\n")
                    f.write(f"入力: {sample['prefix']}\n")
                    f.write(f"生成: {sample['generated']}\n\n")
        except Exception as e:
            print(f"テキスト生成でエラーが発生しました: {e}")
    
    print(f"\n評価結果は {args.output_dir} ディレクトリに保存されました")


if __name__ == "__main__":
    main()
