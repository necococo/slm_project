#!/usr/bin/env python
# coding: utf-8
# slm/main.py - SLM (Simple Language Model)の主要実行スクリプト
# このファイルはWiki40B日本語データセットを使用したDiffusionモデルの学習を行います

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from slm.config import ModelConfig, TrainingConfig, PathsConfig
from slm.modules.wave_network import WaveNetworkLM
from slm.wiki40b_ja_dataset import WikiDataset, collate_fn
# JapaneseTokenizerは使用せず、標準のHuggingFaceトークナイザーのみ使用
from slm.train import Trainer

# TOKENIZERS_PARALLELISMの警告を防ぐために環境変数を設定
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# データセットラッパークラスを追加
class HFDatasetWrapper(Dataset):
    """Hugging Faceデータセットをラップしてcollatorに適した形式を提供するクラス
    
    Notes:
        - WikiDatasetとの互換性を持たせるために必要なインターフェイスを実装
        - 入力データが直接input_ids形式か、textデータのどちらでも対応
        - トークナイザーは標準のHuggingFaceトークナイザー (AutoTokenizer) を使用
    """
    def __init__(self, dataset, tokenizer=None, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer  # トークナイザーを保持
        self.max_length = max_length
        
        # WikiDatasetとの互換性のために属性を追加
        self.file_path = None
        self.memory_efficient = False
        
        # 最初のデータアイテムを確認してデータセットの形式を判断
        if len(dataset) > 0:
            first_item = dataset[0]
            self.has_input_ids = "input_ids" in first_item if isinstance(first_item, dict) else False
            self.has_text = "text" in first_item if isinstance(first_item, dict) else False
            
            # データセットの形式をログ出力
            if isinstance(first_item, dict):
                print(f"データセット形式: {list(first_item.keys())}")
                
                if self.has_text and not self.has_input_ids:
                    if self.tokenizer is None:
                        raise ValueError("テキストデータを処理するにはトークナイザーが必要です")
                    print(f"テキストデータを検出しました。自動的にトークン化します。")
            else:
                print(f"不明なデータセット形式: {type(first_item)}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # collatorが期待する形式にデータを変換
        if isinstance(item, dict):
            # 1. input_idsが直接存在する場合
            if "input_ids" in item:
                input_ids = item["input_ids"]
                attention_mask = item.get("attention_mask", [1] * len(input_ids))
                
                # 最大長を超える場合は切り詰め
                if len(input_ids) > self.max_length:
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                    
                # labelsも含める（train.pyのcollate_fnとの互換性のため）
                result = {
                    "input_ids": input_ids, 
                    "attention_mask": attention_mask
                }
                
                # labelsがない場合はinput_idsをコピー（diffusionモデル用）
                if "labels" not in item:
                    result["labels"] = input_ids.copy() if isinstance(input_ids, list) else input_ids
                else:
                    result["labels"] = item["labels"][:self.max_length]
                    
                return result
                
            # 2. テキストデータがある場合はトークン化
            elif "text" in item and self.tokenizer is not None:
                # トークナイザーを使ってテキストをトークン化
                try:
                    text = item["text"]
                    # AutoTokenizerの場合
                    if hasattr(self.tokenizer, 'encode') and callable(self.tokenizer.encode):
                        if hasattr(self.tokenizer, 'add_special_tokens'):
                            # Transformersトークナイザー
                            input_ids = self.tokenizer.encode(
                                text, 
                                add_special_tokens=False,
                                max_length=self.max_length, 
                                truncation=True
                            )
                        else:
                            # 特殊トークン追加オプションがない場合
                            input_ids = self.tokenizer.encode(text)[:self.max_length]
                    # JapaneseTokenizerのラッパー
                    elif hasattr(self.tokenizer, '_tokenizer'):
                        input_ids = self.tokenizer._tokenizer.encode(
                            text, 
                            add_special_tokens=False,
                            max_length=self.max_length, 
                            truncation=True
                        )
                    # SentencePieceProcessorを使用
                    else:
                        input_ids = self.tokenizer.encode(text)[:self.max_length]
                        
                    # 空のトークン列や失敗した場合の対応
                    if not input_ids:
                        if idx % 1000 == 0:  # エラーログの頻度を減らす
                            print(f"警告: トークン化で空のリストが返されました (idx={idx})")
                        return {"input_ids": [], "attention_mask": [], "labels": []}
                        
                    attention_mask = [1] * len(input_ids)
                    # labelsも含める（diffusionモデル用）
                    return {
                        "input_ids": input_ids, 
                        "attention_mask": attention_mask,
                        "labels": input_ids.copy()
                    }
                except Exception as e:
                    if idx % 1000 == 0:  # エラーログの頻度を減らす
                        print(f"トークン化エラー (idx={idx}): {e}")
                    return {"input_ids": [], "attention_mask": [], "labels": []}
            
            # 3. その他のケース
            else:
                if idx % 1000 == 0:  # 頻度を減らしてログ出力
                    print(f"Warning: Dataset item at index {idx} does not have 'input_ids' or 'text': {item.keys()}")
                
                # 何らかのキーを使って強制的に変換
                if len(item) > 0:
                    first_key = list(item.keys())[0]
                    if isinstance(item[first_key], list):
                        ids = item[first_key][:self.max_length]
                        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids.copy()}
                
                # エラー回避のための空データ
                return {"input_ids": [], "attention_mask": [], "labels": []}
        
        # リスト形式の場合（例えばトークンIDのリスト）
        elif isinstance(item, list):
            ids = item[:self.max_length]
            return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids.copy()}
        
        # テキスト文字列の場合
        elif isinstance(item, str) and self.tokenizer is not None:
            try:
                if hasattr(self.tokenizer, '_tokenizer'):
                    input_ids = self.tokenizer._tokenizer.encode(
                        item, 
                        add_special_tokens=False,
                        max_length=self.max_length, 
                        truncation=True
                    )
                else:
                    input_ids = self.tokenizer.encode(item)[:self.max_length]
                    
                return {
                    "input_ids": input_ids, 
                    "attention_mask": [1] * len(input_ids),
                    "labels": input_ids.copy()
                }
            except Exception as e:
                if idx % 1000 == 0:  # エラーログの頻度を減らす
                    print(f"トークン化エラー (idx={idx}): {e}")
                return {"input_ids": [], "attention_mask": [], "labels": []}
        
        # その他の場合は空の辞書を返す（エラー回避）
        return {"input_ids": [], "attention_mask": [], "labels": []}

def parse_args():
    parser = argparse.ArgumentParser(description="SLM：日本語Wiki40Bデータセットを使用したWave Network+Diffusionモデルの学習")
    
    # データパス関連
    parser.add_argument("--dataset_name", type=str, default="fujiki/wiki40b_ja",
                        help="Hugging Faceからロードするデータセット名")
    parser.add_argument("--local_data_dir", type=str, default="/content/drive/MyDrive/slm/data/fujiki/wiki40b_ja",
                        help="ローカルにダウンロード済みのデータセットディレクトリ")
    parser.add_argument("--fast_data_dir", type=str, default="/content/fast_data",
                        help="高速アクセス用の一時データディレクトリ（デフォルトはランタイム直下）")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="モデル出力ディレクトリ")
    parser.add_argument("--tokenizer_name", type=str, default="megagonlabs/t5-base-japanese-web",
                        help="使用するトークナイザー名")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="保存済みトークナイザーのパス（指定するとHugging Faceからのダウンロードをスキップ）")
    parser.add_argument("--use_local_dataset", action="store_true", default=True,
                        help="ローカルにダウンロード済みのデータセットを使用する")
    parser.add_argument("--use_fast_storage", action="store_true", default=True,
                        help="データをランタイム直下にコピーして高速アクセス（デフォルトはTrue）")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="学習を再開するためのチェックポイントファイルのパス")
    
    # モデル設定
    parser.add_argument("--hidden_size", type=int, default=1024,
                        help="モデルの隠れ層のサイズ")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="モデルのレイヤー数")
    parser.add_argument("--max_seq_len", type=int, default=256,
                        help="最大シーケンス長")
    
    # 学習設定
    parser.add_argument("--batch_size", type=int, default=64,
                        help="バッチサイズ")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Diffusion学習のエポック数")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学習率")
    parser.add_argument("--seed", type=int, default=42,
                        help="乱数シード")
    
    # データセット準備モード
    parser.add_argument("--prepare_datasets", action="store_true",
                        help="データセットの準備のみを行い、トレーニングはスキップする")
    parser.add_argument("--test_samples", type=int, default=1000,
                        help="テストデータセットのサンプル数")
    parser.add_argument("--debug", action="store_true",
                        help="デバッグモード（詳細ログ出力）")
    
    args = parser.parse_args()
    
    # 既存の前処理済みデータが存在すればそれを使う
    # default=Trueに設定済み: args.use_local_dataset = True
    
    # Google Driveのデータディレクトリパスを設定
    args.drive_train_data_dir = os.path.join(args.local_data_dir, "train")
    args.drive_valid_data_dir = os.path.join(args.local_data_dir, "valid")
    args.drive_test_data_dir = os.path.join(args.local_data_dir, "test")
    args.drive_tokenizers_dir = os.path.join(args.local_data_dir, "tokenizers")
    
    # 高速ストレージのデータディレクトリパスを設定
    args.train_data_dir = os.path.join(args.fast_data_dir, "train")
    args.valid_data_dir = os.path.join(args.fast_data_dir, "valid")
    args.test_data_dir = os.path.join(args.fast_data_dir, "test")
    args.tokenizers_dir = os.path.join(args.fast_data_dir, "tokenizers")
    
    # プレーンテキスト用のテストデータ出力先
    args.test_plain_output = os.path.join(args.fast_data_dir, "test_plain.txt")
    
    # 高速ストレージをデフォルトで有効化 (明示的にオフにされない限り)
    if not hasattr(args, 'use_fast_storage') or args.use_fast_storage is None:
        args.use_fast_storage = True
    
    # 高速ストレージのベースディレクトリを作成
    try:
        print(f"高速データディレクトリを作成: {args.fast_data_dir}")
        os.makedirs(args.fast_data_dir, exist_ok=True)
        os.makedirs(args.tokenizers_dir, exist_ok=True)
    except Exception as e:
        print(f"警告: 高速データディレクトリの作成に失敗しました: {e}")
        # 失敗した場合は元のパスを使用
        args.train_data_dir = args.drive_train_data_dir
        args.valid_data_dir = args.drive_valid_data_dir
        args.test_data_dir = args.drive_test_data_dir
        args.tokenizers_dir = args.drive_tokenizers_dir
        args.use_fast_storage = False
    
    return args


def clean_text(batch):
    """テキスト内のプレースホルダートークンを削除し、空行を処理する関数"""
    text = batch["text"]
    
    # プレースホルダートークンの削除
    for token in ["_START_ARTICLE_", "_START_SECTION_", "_START_PARAGRAPH_"]:
        text = text.replace(token, "")
    text = text.replace("_NEWLINE_", "\n")
    
    # 連続する改行を1つに置換
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    
    # 文章間の空行を正規化（2つ以上連続する改行を最大2つに制限）
    lines = text.split("\n")
    non_empty_lines = []
    empty_line_count = 0
    
    for line in lines:
        if line.strip():
            non_empty_lines.append(line)
            empty_line_count = 0
        else:
            # 空行は最大2つまで許可
            if empty_line_count < 2:
                non_empty_lines.append("")
            empty_line_count += 1
    
    # 正規化したテキストを再度結合
    batch["text"] = "\n".join(non_empty_lines)
    return batch


def tokenize_function(examples, tokenizer, max_seq_len, args=None):
    """テキストをトークン化する関数"""
    tokenized = {"input_ids": [], "attention_mask": []}
    
    # テキストキーの存在確認と処理
    if "text" not in examples:
        print(f"警告: 'text'キーが見つかりません。利用可能なキー: {list(examples.keys())}")
        return tokenized
    
    # 空行を含むテキストをフィルタリング
    valid_texts = []
    valid_indices = []
    
    for i, text in enumerate(examples["text"]):
        # 以下の条件のテキストはスキップする:
        # - 空のテキスト
        # - 非常に短いテキスト（10文字未満）
        # - 空行を含むテキスト
        if text and len(text.strip()) >= 10 and "\n\n" not in text:
            valid_texts.append(text)
            valid_indices.append(i)
        elif hasattr(args, 'debug') and args.debug and text:
            reason = "空または短すぎる" if not text or len(text.strip()) < 10 else "空行を含む"
            preview = text[:20] + "..." if len(text) > 20 else text
            print(f"警告: {reason}テキストをスキップします：「{preview}」")
    
    # 有効なテキストだけを処理
    if not valid_texts:
        return {"input_ids": [], "attention_mask": []}
    
    # 有効なテキストのみをトークン化
    for text in valid_texts:
        try:
            # トークン化 - 直接トークナイザーを使用
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            
            # 最大長に切り詰め
            if len(token_ids) > max_seq_len:
                token_ids = token_ids[:max_seq_len]
            
            # 注意マスクを作成（すべて1）
            attn_mask = [1] * len(token_ids)
            
            # 結果を追加
            tokenized["input_ids"].append(token_ids)
            tokenized["attention_mask"].append(attn_mask)
        except Exception as e:
            print(f"トークン化エラー: {e} テキスト: {text[:50]}...")
    
    return tokenized


def prepare_dataset_from_hf(dataset_name, tokenizer, _, max_seq_len, max_valid_samples=1000, args=None):
    """Hugging Faceからデータセットをロードして準備する"""
    print(f"Hugging Faceからデータセット {dataset_name} をロード中...")
    try:
        dataset = load_dataset(dataset_name)
        print(f"データセットのロードに成功しました")
    except Exception as e:
        print(f"エラー: データセットのロードに失敗しました: {e}")
        raise
    
    print(f"データセット情報:")
    for split in dataset:
        print(f"  {split}: {len(dataset[split])}サンプル")
    
    # データセットをトークン化（まずクリーニング）
    print("テキストを前処理中...")
    cleaned_dataset = {}
    for split in dataset:
        cleaned_dataset[split] = dataset[split].map(clean_text)
    
    # テスト、検証セットの自動作成（trainからの分割）
    if "train" in cleaned_dataset and "test" not in cleaned_dataset:
        print("\n検証・テストデータセットがないため、訓練データから自動作成します")
        
        # データセット全体のサイズを取得
        train_size = len(cleaned_dataset["train"])
        test_size = min(args.test_samples if args and hasattr(args, 'test_samples') else 1000, int(train_size * 0.1))
        val_size = min(max_valid_samples, int(train_size * 0.1))
        remaining_size = train_size - test_size - val_size
        
        print(f"データセット分割:")
        print(f"  元の訓練データ: {train_size}サンプル")
        print(f"  新しい訓練データ: {remaining_size}サンプル")
        print(f"  検証データ: {val_size}サンプル")
        print(f"  テストデータ: {test_size}サンプル")
        
        # シード固定でランダムに分割
        generator = torch.Generator().manual_seed(args.seed if args and hasattr(args, 'seed') else 42)
        
        # インデックスをシャッフル
        indices = torch.randperm(train_size, generator=generator).tolist()
        
        # 分割インデックス
        train_indices = indices[:remaining_size]
        val_indices = indices[remaining_size:remaining_size+val_size]
        test_indices = indices[remaining_size+val_size:remaining_size+val_size+test_size]
        
        # 新しいデータセットを作成
        train_part = cleaned_dataset["train"].select(train_indices)
        val_part = cleaned_dataset["train"].select(val_indices)
        test_part = cleaned_dataset["train"].select(test_indices)
        
        # 元のデータセットを置き換え
        cleaned_dataset = {
            "train": train_part,
            "validation": val_part,
            "test": test_part
        }
        
        print("データセットを3分割しました")
        
        # テスト用プレーンテキストも保存（可視化用）- 純粋なテキストのみ
        if args and hasattr(args, 'test_plain_output'):
            print(f"テスト用プレーンテキストデータを保存中: {args.test_plain_output}")
            
            # Google Colab環境で互換性のあるtqdm設定を使用
            from tqdm import tqdm
            with open(args.test_plain_output, "w", encoding="utf-8") as f:
                # プログレスバーの更新頻度を下げる（Colab環境用）
                for item in tqdm(test_part, desc="テキスト保存中", mininterval=0.5):
                    text = item["text"]
                    f.write(text + "\n\n")
            
            print(f"テストサンプル {len(test_part)} 件を保存しました（余計なマークアップなし）")
            
            # トークナイザー情報をargsに保存
            if tokenizer:
                args.tokenizer = tokenizer
    
    # データセットをトークン化
    print("\nデータセットをトークン化中...")
    tokenized_datasets = {}
    for split in cleaned_dataset:
        print(f"  {split}スプリットをトークン化中...")
        tokenized_datasets[split] = cleaned_dataset[split].map(
            lambda examples: tokenize_function(examples, tokenizer, max_seq_len, args),
            batched=True,
            batch_size=1000,
            remove_columns=["text"]
        )
    
    # データセット辞書に変換
    tokenized_datasets = DatasetDict(tokenized_datasets)
    
    print("トークン化済みデータセット情報:")
    for split in tokenized_datasets:
        print(f"  {split}: {len(tokenized_datasets[split])}サンプル")
    
    # データセットの形式を確認
    if "train" in tokenized_datasets and len(tokenized_datasets["train"]) > 0:
        first_item = tokenized_datasets["train"][0]
        print(f"最初のアイテムのキー: {list(first_item.keys())}")
        if "input_ids" in first_item:
            print(f"input_idsの長さ: {len(first_item['input_ids'])}")
            print(f"最初の20トークン: {first_item['input_ids'][:20]}")
        else:
            print(f"警告: 'input_ids'キーがデータセットのアイテムに見つかりません")
    
    # 各スプリットをディレクトリに個別保存
    if args:
        # train
        if "train" in tokenized_datasets:
            try:
                print(f"\nトレーニングデータセットのディレクトリを作成: {args.train_data_dir}")
                os.makedirs(args.train_data_dir, exist_ok=True)
                
                print(f"トレーニングデータセットを保存中... サイズ: {len(tokenized_datasets['train'])}サンプル")
                # DatasetDictではなく直接データセットを保存
                tokenized_datasets["train"].save_to_disk(args.train_data_dir)
                print(f"トレーニングデータの保存が完了しました: {args.train_data_dir}")
            except Exception as e:
                print(f"エラー: トレーニングデータの保存中に問題が発生しました: {e}")
        
        # validation
        if "validation" in tokenized_datasets:
            try:
                print(f"\n検証データセットのディレクトリを作成: {args.valid_data_dir}")
                os.makedirs(args.valid_data_dir, exist_ok=True)
                
                print(f"検証データセットを保存中... サイズ: {len(tokenized_datasets['validation'])}サンプル")
                # DatasetDictではなく直接データセットを保存
                tokenized_datasets["validation"].save_to_disk(args.valid_data_dir)
                print(f"検証データの保存が完了しました: {args.valid_data_dir}")
            except Exception as e:
                print(f"エラー: 検証データの保存中に問題が発生しました: {e}")
        
        # test
        if "test" in tokenized_datasets:
            try:
                print(f"\nテストデータセットのディレクトリを作成: {args.test_data_dir}")
                os.makedirs(args.test_data_dir, exist_ok=True)
                
                print(f"テストデータセットを保存中... サイズ: {len(tokenized_datasets['test'])}サンプル")
                # DatasetDictではなく直接データセットを保存
                tokenized_datasets["test"].save_to_disk(args.test_data_dir)
                print(f"テストデータの保存が完了しました: {args.test_data_dir}")
            except Exception as e:
                print(f"エラー: テストデータの保存中に問題が発生しました: {e}")
            
            # テストデータのプレーンテキスト版も保存 - 純粋なテキストのみ
            if hasattr(args, 'test_plain_output'):
                print(f"テスト用プレーンテキストを保存: {args.test_plain_output}")
                with open(args.test_plain_output, "w", encoding="utf-8") as f:
                    # 純粋なテキストのみ保存（復元後の結果）
                    for i in range(min(len(tokenized_datasets["test"]), 100)):  # サンプル100件のみ表示
                        if i < len(tokenized_datasets["test"]) and "input_ids" in tokenized_datasets["test"][i]:
                            sample_ids = tokenized_datasets["test"][i]["input_ids"]
                            sample_text = tokenizer.decode(sample_ids) if tokenizer else f"{sample_ids[:50]}"
                            f.write(f"{sample_text}\n\n")
        
        print("\nデータセットの保存が完了しました。各スプリットは以下のディレクトリにあります:")
        print(f"  train: {args.train_data_dir}")
        print(f"  validation: {args.valid_data_dir}")
        print(f"  test: {args.test_data_dir}")
        print("\n利用する場合は --use_local_dataset --local_data_dir=\"{親ディレクトリ}\" を指定してください。")
    
    return tokenized_datasets




# 標準のHuggingFaceトークナイザーのロード関数はslm/tokenizer.pyに移動しました


def main():
    # コマンドライン引数を解析
    args = parse_args()
    
    # データセット準備モードの場合の特別メッセージ
    if args.prepare_datasets:
        print("==== データセット準備モード ====")
        print("データセットの準備はCPUのみで実行されます。GPUは不要です。")
    
    # 乱数シード設定
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Google Colab環境の検出
    try:
        import google.colab
        is_colab = True
        print("Google Colab環境で実行中")

        # Colabでのtqdmの設定を最適化
        # ウィジェットではなくテキストベースのプログレスバーを使用
        import tqdm
        # tqdm設定を変更してColab環境に最適化
        # 1. 更新頻度を下げる
        tqdm.tqdm.mininterval = 0.5  # 最小更新間隔を0.5秒に設定
        # 2. デフォルトでリフレッシュレートを低く設定
        tqdm.tqdm.default_mininterval = 0.5
    except ImportError:
        is_colab = False
        print("ローカル環境で実行中")
    
    # トークナイザーのロード
    from slm.tokenizer import load_tokenizer
    
    # トークナイザーの保存先は常に高速ストレージ内
    tokenizer_save_dir = args.tokenizers_dir
    
    # 高速ストレージディレクトリが存在することを確認
    os.makedirs(tokenizer_save_dir, exist_ok=True)

    # トークナイザーの検索順序を決定
    tokenizer_path = None
    
    # 既存の tokenizer_path パラメータがあればそれを優先
    if args.tokenizer_path and os.path.exists(os.path.dirname(args.tokenizer_path)):
        tokenizer_path = args.tokenizer_path
    else:
        # 検索順序: 1. 高速ストレージ、2. ローカルパス、3. その他の既定のパス
        search_paths = []
        
        # 1. 高速ストレージを常に最優先
        search_paths.extend([
            os.path.join(args.tokenizers_dir, "tokenizer.pkl"),
            os.path.join(args.tokenizers_dir, "tokenizer_model.json")
        ])
        
        # 2. ローカルディレクトリパス
        search_paths.extend([
            os.path.join(args.drive_tokenizers_dir, "tokenizer.pkl"),
            os.path.join(args.drive_tokenizers_dir, "tokenizer_model.json"),
            os.path.join("/content/drive/MyDrive/slm/checkpoints/tokenizers", "tokenizer_model.json")
        ])
        
        # 検索実行
        for path in search_paths:
            if os.path.exists(path):
                tokenizer_path = path
                location = "高速ストレージ" if args.tokenizers_dir in path else "ローカルディレクトリ"
                print(f"{location}で既存のトークナイザーを発見しました: {tokenizer_path}")
                break
    
    # ローカルからトークナイザーをコピー (高速ストレージになければ)
    if tokenizer_path and not tokenizer_path.startswith(args.tokenizers_dir):
        try:
            import shutil
            # コピー元のディレクトリまたはファイル
            source = os.path.dirname(tokenizer_path) if os.path.isfile(tokenizer_path) else tokenizer_path
            
            print(f"トークナイザーを高速ストレージにコピー中: {source} → {args.tokenizers_dir}")
            
            # ディレクトリ全体をコピー
            if os.path.isdir(source):
                shutil.copytree(source, args.tokenizers_dir, dirs_exist_ok=True)
            # 単一ファイルのコピー
            else:
                target_file = os.path.join(args.tokenizers_dir, os.path.basename(tokenizer_path))
                shutil.copy2(tokenizer_path, target_file)
                
            print(f"トークナイザーを高速ストレージにコピーしました")
            
            # 高速ストレージ内のパスを優先するよう更新
            if tokenizer_path.endswith("tokenizer.pkl"):
                tokenizer_path = os.path.join(args.tokenizers_dir, "tokenizer.pkl")
            elif tokenizer_path.endswith("tokenizer_model.json"):
                tokenizer_path = os.path.join(args.tokenizers_dir, "tokenizer_model.json")
            
        except Exception as e:
            print(f"トークナイザーの高速ストレージへのコピーに失敗しました: {e}")
    
    # トークナイザーを読み込む
    print(f"トークナイザーを読み込み中...")
    try:
        # 拡張された load_tokenizer 関数を使用
        tokenizer = load_tokenizer(
            model_name=args.tokenizer_name,
            tokenizer_path=tokenizer_path,
            save_dir=tokenizer_save_dir,
            use_fast=False
        )
        
        # 常に高速ストレージにPickle形式でも保存
        try:
            import pickle
            pickle_path = os.path.join(args.tokenizers_dir, "tokenizer.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(tokenizer, f)
            print(f"トークナイザーをPickle形式でも保存しました: {pickle_path}")
        except Exception as e:
            print(f"Pickle形式での保存に失敗しました: {e}")
            
    except Exception as e:
        print(f"トークナイザーのロードに失敗しました: {e}")
        print(f"標準のHugging Faceトークナイザーを使用します")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)
        
        # 新しくダウンロードしたトークナイザーは高速ストレージに保存
        try:
            tokenizer.save_pretrained(args.tokenizers_dir)
            print(f"ダウンロードしたトークナイザーを高速ストレージに保存しました: {args.tokenizers_dir}")
            
            # Pickleでも保存
            import pickle
            pickle_path = os.path.join(args.tokenizers_dir, "tokenizer.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(tokenizer, f)
            print(f"トークナイザーをPickle形式でも保存しました: {pickle_path}")
        except Exception as e:
            print(f"トークナイザーの保存に失敗しました: {e}")
    
    mask_token_id = tokenizer.mask_token_id
    print(f"[MASK]トークンID: {mask_token_id}")
    
    # データセット準備の場合はハイスピードモードを無効化
    if args.prepare_datasets:
        args.use_fast_storage = False
        print("データセット準備モードが有効なため、ハイスピードモードを無効化します。")
    
    # データセットの情報表示
    print(f"\nデータディレクトリ情報:")
    if args.use_fast_storage:
        print(f"  元のデータディレクトリ: {args.local_data_dir}")
        print(f"  高速アクセス用ディレクトリ: {args.fast_data_dir}")
        print(f"  - 訓練: {args.train_data_dir}")
        print(f"  - 検証: {args.valid_data_dir}")
        print(f"  - テスト: {args.test_data_dir}")
    else:
        print(f"  データディレクトリ: {args.local_data_dir}")
        print(f"  - 訓練: {args.drive_train_data_dir}")
        print(f"  - 検証: {args.drive_valid_data_dir}")
        print(f"  - テスト: {args.drive_test_data_dir}")
    
    # 常に高速ストレージを使用
    # トレーニングデータが高速ストレージに存在するかチェック
    fast_data_exists = os.path.exists(args.train_data_dir)
    
    if not fast_data_exists:
        # サブディレクトリを明示的に作成
        try:
            print("\n高速ストレージにデータをコピーします...")
            
            # 各ディレクトリを作成
            os.makedirs(args.train_data_dir, exist_ok=True)
            os.makedirs(args.valid_data_dir, exist_ok=True)
            os.makedirs(args.test_data_dir, exist_ok=True)
            
            # Google Driveから高速ストレージにデータをコピー
            import shutil
            from datetime import datetime
            import time
            
            # コピー開始時間
            start_time = datetime.now()
            print(f"コピー開始: {start_time.strftime('%H:%M:%S')}")
            
            # 必要なディレクトリのみコピー（訓練データが最優先）
            copied_dirs = []
            
            # 訓練データは必須
            train_data_available = os.path.exists(args.drive_train_data_dir)
            if not train_data_available:
                print(f"エラー: 訓練データが見つかりません: {args.drive_train_data_dir}")
                # 警告を表示するが、引き続き高速ストレージを使用
                print(f"警告: 元データが見つからないため、高速ストレージにデータがない状態で続行します。")
            else:
                print(f"訓練データをコピー中: {args.drive_train_data_dir} → {args.train_data_dir}")
                copy_start = time.time()
                shutil.copytree(args.drive_train_data_dir, args.train_data_dir, dirs_exist_ok=True)
                copy_end = time.time()
                copied_dirs.append(f"train ({copy_end - copy_start:.1f}秒)")
            
                # 検証データのコピー（あれば）
                if os.path.exists(args.drive_valid_data_dir):
                    print(f"検証データをコピー中: {args.drive_valid_data_dir} → {args.valid_data_dir}")
                    copy_start = time.time()
                    shutil.copytree(args.drive_valid_data_dir, args.valid_data_dir, dirs_exist_ok=True)
                    copy_end = time.time()
                    copied_dirs.append(f"valid ({copy_end - copy_start:.1f}秒)")
                else:
                    print(f"検証データが見つかりません。スキップします: {args.drive_valid_data_dir}")
                
                # テストデータのコピー（あれば）
                if os.path.exists(args.drive_test_data_dir):
                    print(f"テストデータをコピー中: {args.drive_test_data_dir} → {args.test_data_dir}")
                    copy_start = time.time()
                    shutil.copytree(args.drive_test_data_dir, args.test_data_dir, dirs_exist_ok=True)
                    copy_end = time.time()
                    copied_dirs.append(f"test ({copy_end - copy_start:.1f}秒)")
                
                # トークナイザーのコピー（あれば、既にトークナイザー処理でコピー済みでない場合）
                if os.path.exists(args.drive_tokenizers_dir) and not os.path.exists(os.path.join(args.tokenizers_dir, "tokenizer.pkl")):
                    print(f"トークナイザーをコピー中: {args.drive_tokenizers_dir} → {args.tokenizers_dir}")
                    copy_start = time.time()
                    os.makedirs(args.tokenizers_dir, exist_ok=True)
                    shutil.copytree(args.drive_tokenizers_dir, args.tokenizers_dir, dirs_exist_ok=True)
                    copy_end = time.time()
                    copied_dirs.append(f"tokenizers ({copy_end - copy_start:.1f}秒)")
            
                # コピー終了時間
                end_time = datetime.now()
                elapsed = end_time - start_time
                print(f"コピー完了: {end_time.strftime('%H:%M:%S')} (所要時間: {elapsed.total_seconds():.1f}秒)")
                print(f"コピーしたディレクトリ: {', '.join(copied_dirs)}")
                print(f"高速アクセスのためのデータを /content/fast_dataへのコピーが完了しました")
            
        except Exception as e:
            print(f"エラー: 高速ストレージへのデータコピーに失敗しました: {e}")
            print(f"警告: コピーに失敗しましたが、引き続き高速ストレージを使用します。")
            print(f"必要に応じてHugging Faceからデータセットをダウンロードし、高速ストレージに保存します。")
    else:
        print(f"高速ストレージにデータが既に存在します: {args.train_data_dir}")
        
    # 常に高速ストレージのパスを使用
    print(f"高速データアクセスを使用: {args.fast_data_dir}")
    
    # デバッグモードの場合、詳細情報を表示
    if hasattr(args, 'debug') and args.debug:
        print(f"\nディレクトリ詳細:")
        print(f"  訓練データディレクトリの絶対パス: {os.path.abspath(args.train_data_dir)}")
        print(f"  訓練データディレクトリ存在: {os.path.exists(args.train_data_dir)}")
    
    if args.use_local_dataset:
        print(f"\nローカルデータセットを使用します")
        # ローカルデータセットの読み込み
        dataset = {}
        
        # 各スプリットを読み込み
        try:
            # 最も柔軟に対応できるよう、各ディレクトリの存在をチェック
            
            if os.path.exists(args.train_data_dir) and os.path.exists(args.valid_data_dir) and os.path.exists(args.test_data_dir):
                print("スプリット別ディレクトリ形式のデータセットを読み込み中...")
                dataset = {}
                
                # 各スプリットを個別に読み込み
                if os.path.exists(args.train_data_dir):
                    try:
                        train_dataset = load_from_disk(args.train_data_dir)
                        dataset["train"] = train_dataset["train"] if "train" in train_dataset else train_dataset
                        
                        print(f"訓練データを読み込みました: {len(dataset['train'])} 件")
                    except Exception as e:
                        print(f"訓練データの読み込みに失敗: {e}")
                
                if os.path.exists(args.valid_data_dir):
                    try:
                        valid_dataset = load_from_disk(args.valid_data_dir)
                        dataset["validation"] = valid_dataset["validation"] if "validation" in valid_dataset else valid_dataset
                        print(f"検証データを読み込みました: {len(dataset['validation'])} 件")
                    except Exception as e:
                        print(f"検証データの読み込みに失敗: {e}")
                
                if os.path.exists(args.test_data_dir):
                    try:
                        test_dataset = load_from_disk(args.test_data_dir)
                        dataset["test"] = test_dataset["test"] if "test" in test_dataset else test_dataset
                        print(f"テストデータを読み込みました: {len(dataset['test'])} 件")
                    except Exception as e:
                        print(f"テストデータの読み込みに失敗: {e}")
                
                if not dataset:
                    raise ValueError("データセットが空です")
                print("利用可能なスプリット:", list(dataset.keys()))
            
            # 3. 部分的なスプリット（訓練データのみがあればOK）
            elif os.path.exists(args.train_data_dir):
                print("訓練データのみのディレクトリ形式を読み込み中...")
                dataset = {}
                
                try:
                    train_dataset = load_from_disk(args.train_data_dir)
                    dataset["train"] = train_dataset["train"] if "train" in train_dataset else train_dataset
                    print(f"訓練データを読み込みました: {len(dataset['train'])} 件")
                except Exception as e:
                    print(f"訓練データの読み込みに失敗: {e}")
                    raise
                    
                print("検証データとテストデータはありません。必要に応じて自動的に作成されます。")
            else:
                # 前処理済みデータがない場合のみ新しくデータセット作成
                print(f"ディレクトリ構造が想定と異なります: {args.local_data_dir}")
                print("Hugging Faceからデータセットをロードします...")
                dataset = prepare_dataset_from_hf(args.dataset_name, tokenizer, tokenizer, 
                                                args.max_seq_len, max_valid_samples=1000, args=args)
        except Exception as e:
            print(f"ローカルデータセットのロードに失敗しました: {e}")
            print("Hugging Faceからデータセットをロードします...")
            dataset = prepare_dataset_from_hf(args.dataset_name, tokenizer, tokenizer, 
                                            args.max_seq_len, max_valid_samples=1000, args=args)
    else:
        # Hugging Faceからデータセットを準備
        print("Hugging Faceからデータセットをロードして準備します")
        dataset = prepare_dataset_from_hf(args.dataset_name, tokenizer, tokenizer, 
                                         args.max_seq_len, max_valid_samples=1000, args=args)
        
        # 将来の使用のためにローカルに保存は prepare_dataset_from_hf 内で行われる
    
    # データセット準備のみのモードの場合は終了
    if args.prepare_datasets:
        print("データセット準備モードが有効なため、トレーニングをスキップします。")
        print("\n==== データセット準備完了 ====")
        print(f"データディレクトリ: {args.local_data_dir}")
        print(f"  - 訓練データ: {args.train_data_dir}")
        print(f"  - 検証データ: {args.valid_data_dir}")
        print(f"  - テストデータ: {args.test_data_dir}")
        
        if "train" in dataset:
            print(f"訓練データサンプル数: {len(dataset['train'])}")
        if "validation" in dataset:
            print(f"検証データサンプル数: {len(dataset['validation'])}")
        if "test" in dataset:
            print(f"テストデータサンプル数: {len(dataset['test'])}")
            
        print("\n利用方法:")
        print(f"  python slm/main.py \\")
        print(f"      --use_local_dataset \\")
        print(f"      --local_data_dir=\"{args.local_data_dir}\" \\")
        print(f"      --output_dir=\"/path/to/output\" \\")
        print(f"      --batch_size=8 \\")
        print(f"      --epochs=3")
        return
    
    # 語彙サイズを安全に取得
    if hasattr(tokenizer, 'vocab_size'):
        vocab_size = tokenizer.vocab_size
    elif hasattr(tokenizer, 'vocab') and isinstance(tokenizer.vocab, dict):
        vocab_size = len(tokenizer.vocab)
    else:
        # 安全のためのデフォルト値
        vocab_size = 32100  # megagonlabs/t5-base-japanese-webの語彙サイズ
        print(f"警告: トークナイザーから語彙サイズを取得できませんでした。デフォルト値 {vocab_size} を使用します。")
    
    print(f"使用する語彙サイズ: {vocab_size}")
    
    # mask_token_idがvocab_sizeを超えている場合の対応
    # 既存モデルとの互換性のため、mask_token_idを変更せず、vocab_sizeを拡張する方針に変更
    if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
        print(f"マスクトークンID: {tokenizer.mask_token_id}")
        # まず現在の語彙サイズに余裕をもたせる（mask_token_id + 1で確実に含まれるように）
        if tokenizer.mask_token_id >= vocab_size:
            old_vocab_size = vocab_size
            vocab_size = tokenizer.mask_token_id + 1  # 確実にmask_token_idが含まれるサイズに
            print(f"警告: マスクトークンID ({tokenizer.mask_token_id}) が語彙サイズ ({old_vocab_size}) を超えています")
            print(f"既存モデルとの互換性を維持するため、vocab_sizeを {old_vocab_size} から {vocab_size} に拡張しました")
            print(f"注意: mask_token_idは変更せず {tokenizer.mask_token_id} のまま使用します")
    
    # モデル設定
    print("\nモデルを初期化します...")
    model_config = ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        vocab_size=vocab_size,  # 拡張後のvocab_sizeを使用
        max_seq_len=args.max_seq_len,
        dropout_prob=0.2,
        use_rope=True,
        use_wavelet=True,
        wavelet_name="haar",
        activation="gelu",
        use_bio_noise=True,
        noise_std=0.1
    )
    
    # トークナイザーをモデル設定に追加
    model_config.set_tokenizer(tokenizer)
    
    # トレーニング設定
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,  # 元のコマンドライン引数の値を使用
        batch_size=args.batch_size,  # 元のコマンドライン引数の値を使用
        mlm_epochs=0,  # MLM学習はスキップ
        diffusion_epochs=args.epochs,  # Diffusion学習のみ実行
        weight_decay=0.01,
        warmup_steps=500,
        use_amp=True,
        use_gradient_checkpointing=True,
        clip_grad_norm=True,
        clip_value=0.5  # 安定性のため勾配クリッピング値を小さくする
    )
    
    # パス設定
    paths_config = PathsConfig(
        base_dir=os.getcwd(),
        output_dir=args.output_dir,
        run_name=f"slm_{args.hidden_size}h_{args.num_layers}l"
    )
    
    # Google Driveの検出（Colabの場合）
    if is_colab and "/content/drive/" in args.output_dir:
        print("Google Driveが検出されました。結果をGoogle Driveに保存します。")
        print(f"実行ID: {paths_config.run_name}")
        print(f"出力ディレクトリ: {args.output_dir}")
    
    # モデルのインスタンス化
    model = WaveNetworkLM(model_config)
    
    # モデル情報の出力
    param_count = sum(p.numel() for p in model.parameters())
    print(f"モデルを初期化しました。パラメータ数: {param_count:,}")
    
    # トークナイザーの動作確認
    print("\n=== トークナイザーの動作確認 ===")
    test_text = "これはトークナイザーのテストです。日本語Wikipediaで学習されたモデルを使います。"
    print(f"テスト文: {test_text}")
    
    # トークン化 - 標準のトークナイザーでテスト
    tokens_ids = tokenizer.encode(test_text, add_special_tokens=False)
    
    # トークン情報の出力
    print(f"トークンID: {tokens_ids}")
    
    # トークン文字列の表示
    tokens_str = tokenizer.convert_ids_to_tokens(tokens_ids)
    print(f"トークン文字列: {tokens_str}")
    
    # デコードのテスト
    decoded_text = tokenizer.decode(tokens_ids, skip_special_tokens=True)
    print(f"デコード結果: {decoded_text}")
    
    # 特殊トークンの確認
    print(f"\n特殊トークン情報:")
    print(f"  MASK: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
    print(f"  語彙サイズ: {tokenizer.vocab_size}")
    
    # トレーニングデータの最初のバッチをチェック
    print("\n=== トレーニングデータのサンプル ===")
    if "train" in dataset:
        # 最初のサンプルを取得
        sample = dataset["train"][0]
        print(f"サンプルの形状: {len(sample['input_ids']) if 'input_ids' in sample else 'Unknown'}")
        
        if 'input_ids' in sample:
            # 最初のサンプルのトークンIDを表示
            sample_ids = sample['input_ids']
            print(f"サンプルのトークンID (最初の20個): {sample_ids[:20]}")
            
            # トークンIDをデコードして表示
            sample_text = tokenizer.decode(sample_ids, skip_special_tokens=True)
            print(f"サンプルテキスト: {sample_text[:100]}..." if len(sample_text) > 100 else sample_text)
    else:
        print("トレーニングデータセットが見つかりません")
    
    # Colab環境の場合、メモリ使用量や各種設定を調整
    if is_colab:
        # バッチサイズの調整
        # original_batch_size = training_config.batch_size
        # training_config.batch_size = min(training_config.batch_size, 4)
        # if original_batch_size != training_config.batch_size:
        #     print(f"Colab環境向けにバッチサイズを調整: {original_batch_size} → {training_config.batch_size}")
        
        # AMP（Automatic Mixed Precision）の確認
        if hasattr(training_config, 'use_amp') and training_config.use_amp:
            if not torch.cuda.is_available():
                print("GPUが利用できないため、AMPを無効にします")
                training_config.use_amp = False
        
        print("Google Colab環境向けに設定を最適化しました")
    
    # データセットのキーをチェック
    print(f"データセットのキー: {list(dataset.keys() if hasattr(dataset, 'keys') else [])}")
    if "train" in dataset:
        print(f"トレーニングデータセットの最初の項目のキー: {list(dataset['train'][0].keys())}")
    
    # データセット形式の確認
    print("\n=== データセット形式の確認 ===")
    if "train" in dataset and len(dataset["train"]) > 0:
        first_item = dataset["train"][0]
        if isinstance(first_item, dict):
            print(f"最初のトレーニングアイテムのキー: {list(first_item.keys())}")
            if "text" in first_item and "input_ids" not in first_item:
                print("データセットには 'text' キーのみが含まれています。自動的にトークン化を行います。")
            elif "input_ids" in first_item:
                print("データセットには既に 'input_ids' が含まれています。")
        else:
            print(f"予期しない形式のデータセット: {type(first_item)}")
    
    # トークナイザーが正しく動作することを確認
    tokenizer_for_wrapper = tokenizer  # デフォルトはロードしたトークナイザー
    
    # サンプルテキストでトークナイザーをテスト
    test_text = "これはトークナイザーのテストです。"
    try:
        test_tokens = tokenizer.encode(test_text, add_special_tokens=False)
        print(f"トークナイザーのテスト結果: {test_tokens[:10]}...")
    except Exception as e:
        print(f"トークナイザーエラー: {e}")
        print("警告: トークナイザーが正常に動作しません。")
    
    # 検証データセットとテストデータセットは、prepare_dataset_from_hf関数内で
    # 自動的に作成されるため、ここでの処理は不要になりました
    
    # データセットをラッパーでラップしてcollatorに適した形式に変換
    # トークナイザーも渡して自動トークン化できるようにする
    train_dataset = HFDatasetWrapper(
        dataset["train"], 
        tokenizer=tokenizer_for_wrapper,  # テスト済みトークナイザーを使用
        max_length=args.max_seq_len
    ) if "train" in dataset else None
    
    valid_dataset = HFDatasetWrapper(
        dataset["validation"], 
        tokenizer=tokenizer_for_wrapper,  # テスト済みトークナイザーを使用
        max_length=args.max_seq_len
    ) if "validation" in dataset else None
    
    # トレーナーの初期化
    try:
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            training_config=training_config,
            paths_config=paths_config,
            seed=args.seed
        )
        
        # チェックポイントからモデルを読み込む（指定されている場合）
        start_epoch = 0
        if args.checkpoint:
            print(f"\n==== チェックポイントから学習を再開します ====")
            print(f"チェックポイントパス: {args.checkpoint}")
            start_epoch, start_step = trainer.load_checkpoint(args.checkpoint)
            print(f"チェックポイントの読み込みが完了しました。エポック {start_epoch} から再開します。")
        
        # 訓練の実行前に重要なメッセージを表示
        print("\n==== 重要: データセット情報 ====")
        print(f"訓練データ件数: {'不明' if 'train' not in dataset else len(dataset['train'])} 件")
        print(f"検証データ件数: {'不明' if 'validation' not in dataset else len(dataset['validation'])} 件")
        print(f"HFDatasetWrapperでラップ済み: {isinstance(train_dataset, HFDatasetWrapper)}")
        print(f"GPU/TPU利用可能: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Diffusion訓練の実行
        print("\nDiffusionモデルの学習を開始します...")
        trainer.train_diffusion(start_epoch=start_epoch, start_step=start_step)
        
        # 最終チェックポイントの保存
        trainer.save_checkpoint("final_model")
        
        # リソースの解放
        trainer.close()
        
        print("学習が完了しました。最終モデルが保存されました。\n")
        print("注意: このバージョンでは、データセットは以下のディレクトリ構造で保存されます:")
        print(f"  {args.local_data_dir}/")
        print(f"  ├── train/  # 訓練データ")
        print(f"  ├── valid/  # 検証データ")
        print(f"  └── test/   # テストデータ")
    except Exception as e:
        print(f"トレーニング中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()