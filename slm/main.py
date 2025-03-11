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
from slm.tokenizer import JapaneseTokenizer
from slm.train import Trainer

# TOKENIZERS_PARALLELISMの警告を防ぐために環境変数を設定
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# データセットラッパークラスを追加
class HFDatasetWrapper(Dataset):
    """Hugging Faceデータセットをラップしてcollatorに適した形式を提供するクラス
    
    Notes:
        - WikiDatasetとの互換性を持たせるために必要なインターフェイスを実装
        - 入力データが直接input_ids形式か、textデータのどちらでも対応
        - トークナイザーはhf_tokenizer (transformers.AutoTokenizer) または
          JapaneseTokenizerどちらも受け付ける
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
    parser.add_argument("--dataset_name", type=str, default="toramaru-u/wiki40b-ja",
                        help="Hugging Faceからロードするデータセット名")
    parser.add_argument("--local_data_dir", type=str, default="./data/wiki40b_ja",
                        help="ローカルにダウンロード済みのデータセットディレクトリ")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="モデル出力ディレクトリ")
    parser.add_argument("--tokenizer_name", type=str, default="megagonlabs/t5-base-japanese-web",
                        help="使用するトークナイザー名")
    parser.add_argument("--use_local_dataset", action="store_true",
                        help="ローカルにダウンロード済みのデータセットを使用する")
    
    # モデル設定
    parser.add_argument("--hidden_size", type=int, default=1024,
                        help="モデルの隠れ層のサイズ")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="モデルのレイヤー数")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="最大シーケンス長")
    
    # 学習設定
    parser.add_argument("--batch_size", type=int, default=8,
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
    
    args = parser.parse_args()
    
    # データセットのスプリット別サブディレクトリを作成
    args.train_data_dir = os.path.join(args.local_data_dir, "train")
    args.valid_data_dir = os.path.join(args.local_data_dir, "valid")
    args.test_data_dir = os.path.join(args.local_data_dir, "test")
    
    # プレーンテキスト用のテストデータ出力先も設定
    args.test_plain_output = os.path.join(args.local_data_dir, "test_plain.txt")
    
    return args


def clean_text(batch):
    """テキスト内のプレースホルダートークンを削除する関数"""
    text = batch["text"]
    for token in ["_START_ARTICLE_", "_START_SECTION_", "_START_PARAGRAPH_"]:
        text = text.replace(token, "")
    text = text.replace("_NEWLINE_", "\n")
    batch["text"] = text
    return batch


def tokenize_function(examples, hf_tokenizer, max_seq_len):
    """テキストをトークン化する関数"""
    tokenized = {"input_ids": [], "attention_mask": []}
    
    # テキストキーの存在確認と処理
    if "text" not in examples:
        print(f"警告: 'text'キーが見つかりません。利用可能なキー: {list(examples.keys())}")
        return tokenized
    
    for text in examples["text"]:
        if not text:  # 空のテキストはスキップ
            tokenized["input_ids"].append([])
            tokenized["attention_mask"].append([])
            continue
            
        try:
            # トークン化 - 直接Hugging Faceトークナイザーを使用
            token_ids = hf_tokenizer.encode(text, add_special_tokens=False)
            
            # 最大長に切り詰め
            if len(token_ids) > max_seq_len:
                token_ids = token_ids[:max_seq_len]
            
            # 注意マスクを作成（すべて1）
            attn_mask = [1] * len(token_ids)
        except Exception as e:
            print(f"トークン化エラー: {e} テキスト: {text[:50]}...")
            # エラー時は空のリストを追加
            token_ids = []
            attn_mask = []
        
        tokenized["input_ids"].append(token_ids)
        tokenized["attention_mask"].append(attn_mask)
    
    return tokenized


def prepare_dataset_from_hf(dataset_name, tokenizer, hf_tokenizer, max_seq_len, max_valid_samples=1000, args=None):
    """Hugging Faceからデータセットをロードして準備する"""
    print(f"Hugging Faceからデータセット {dataset_name} をロード中...")
    dataset = load_dataset(dataset_name)
    
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
            
            # トークナイザーも保存
            if hf_tokenizer:
                args.hf_tokenizer = hf_tokenizer
    
    # データセットをトークン化
    print("\nデータセットをトークン化中...")
    tokenized_datasets = {}
    for split in cleaned_dataset:
        print(f"  {split}スプリットをトークン化中...")
        tokenized_datasets[split] = cleaned_dataset[split].map(
            lambda examples: tokenize_function(examples, hf_tokenizer, max_seq_len),
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
            os.makedirs(args.train_data_dir, exist_ok=True)
            print(f"\nトレーニングデータセットを保存: {args.train_data_dir}")
            train_dataset_dict = DatasetDict({"train": tokenized_datasets["train"]})
            train_dataset_dict.save_to_disk(args.train_data_dir)
        
        # validation
        if "validation" in tokenized_datasets:
            os.makedirs(args.valid_data_dir, exist_ok=True)
            print(f"検証データセットを保存: {args.valid_data_dir}")
            valid_dataset_dict = DatasetDict({"validation": tokenized_datasets["validation"]})
            valid_dataset_dict.save_to_disk(args.valid_data_dir)
        
        # test
        if "test" in tokenized_datasets:
            os.makedirs(args.test_data_dir, exist_ok=True)
            print(f"テストデータセットを保存: {args.test_data_dir}")
            test_dataset_dict = DatasetDict({"test": tokenized_datasets["test"]})
            test_dataset_dict.save_to_disk(args.test_data_dir)
            
            # テストデータのプレーンテキスト版も保存 - 純粋なテキストのみ
            if hasattr(args, 'test_plain_output'):
                print(f"テスト用プレーンテキストを保存: {args.test_plain_output}")
                with open(args.test_plain_output, "w", encoding="utf-8") as f:
                    # 純粋なテキストのみ保存（復元後の結果）
                    for i in range(min(len(tokenized_datasets["test"]), 100)):  # サンプル100件のみ表示
                        if i < len(tokenized_datasets["test"]) and "input_ids" in tokenized_datasets["test"][i]:
                            sample_ids = tokenized_datasets["test"][i]["input_ids"]
                            if hasattr(args, 'hf_tokenizer'):
                                sample_text = args.hf_tokenizer.decode(sample_ids)
                            else:
                                sample_text = f"{sample_ids[:50]}"
                            f.write(f"{sample_text}\n\n")
        
        print("\nデータセットの保存が完了しました。各スプリットは以下のディレクトリにあります:")
        print(f"  train: {args.train_data_dir}")
        print(f"  validation: {args.valid_data_dir}")
        print(f"  test: {args.test_data_dir}")
        print("\n利用する場合は --use_local_dataset --local_data_dir=\"{親ディレクトリ}\" を指定してください。")
    
    return tokenized_datasets




def load_tokenizer_megagon(tokenizer_name):
    """megagonlabs/t5-base-japanese-webなどのトークナイザーをロード"""
    print(f"トークナイザー {tokenizer_name} をロード中...")
    
    # まずHuggingFaceからロード
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    
    # T5トークナイザーにはデフォルトのマスクトークンがないため、必要に応じて追加
    if not hasattr(hf_tokenizer, 'mask_token') or hf_tokenizer.mask_token is None:
        # マスクトークンの追加
        hf_tokenizer.add_special_tokens({'mask_token': '<mask>'})
        print(f"マスクトークン '<mask>' を追加しました。")
    else:
        print(f"既存のマスクトークン: {hf_tokenizer.mask_token}")
    
    # BOSトークンの追加（必要に応じて）
    if not hasattr(hf_tokenizer, 'bos_token') or hf_tokenizer.bos_token is None:
        hf_tokenizer.add_special_tokens({'bos_token': '<s>'})
        print(f"BOSトークン '<s>' を追加しました。")
    else:
        print(f"既存のBOSトークン: {hf_tokenizer.bos_token}")
    
    # マスクトークンIDを確認
    mask_token_id = hf_tokenizer.mask_token_id
    print(f"マスクトークンID: {mask_token_id}")
    
    # JapaneseTokenizerラッパーに変換
    jp_tokenizer = JapaneseTokenizer.from_pretrained_tokenizer(hf_tokenizer)
    
    print(f"トークナイザーをロードしました。語彙サイズ: {len(hf_tokenizer.vocab) if hasattr(hf_tokenizer, 'vocab') else hf_tokenizer.vocab_size}")
    
    # トークナイザーの設定をJapaneseTokenizerに反映（マスクトークンなど）
    jp_tokenizer.mask_token = hf_tokenizer.mask_token
    jp_tokenizer.mask_token_id = hf_tokenizer.mask_token_id
    print(f"JapaneseTokenizerのマスクトークン: {jp_tokenizer.mask_token}, ID: {jp_tokenizer.mask_token_id}")
    
    return jp_tokenizer, hf_tokenizer


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
        
        # Colabでのプログレスバー設定の最適化
        # 入力プロンプトを防ぐため、ウィジェットの特別設定
        try:
            from google.colab import output
            output.enable_custom_widget_manager()
            print("Colabウィジェットマネージャーを有効化しました")
        except:
            print("Colabウィジェットマネージャーの有効化に失敗しました")
        
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
    tokenizer, hf_tokenizer = load_tokenizer_megagon(args.tokenizer_name)
    mask_token_id = tokenizer.mask_token_id
    
    print(f"[MASK]トークンID: {mask_token_id}")
    
    # データセット準備では、トレーニング・テスト・検証データが自動的に作成されます
    
    # データセットの準備
    print(f"データディレクトリ: {args.local_data_dir}")
    print(f"  - 訓練: {args.train_data_dir}")
    print(f"  - 検証: {args.valid_data_dir}")
    print(f"  - テスト: {args.test_data_dir}")
    
    if args.use_local_dataset:
        print(f"ローカルデータセットを使用します")
        # ローカルデータセットの読み込み
        dataset = {}
        
        # 各スプリットを読み込み
        try:
            # 最も柔軟に対応できるよう、各ディレクトリの存在をチェック
            
            # 1. 旧フォーマット（単一ディレクトリ）
            if os.path.exists(os.path.join(args.local_data_dir, "dataset_info.json")):
                print("単一ディレクトリ形式のデータセットを読み込み中...")
                full_dataset = load_from_disk(args.local_data_dir)
                dataset = full_dataset
                print("データセットを正常にロードしました")
                
            # 2. 新フォーマット（完全なスプリット別ディレクトリ）
            elif os.path.exists(args.train_data_dir) and os.path.exists(args.valid_data_dir) and os.path.exists(args.test_data_dir):
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
                raise FileNotFoundError(f"ディレクトリ構造が想定と異なります: {args.local_data_dir}")
        except Exception as e:
            print(f"ローカルデータセットのロードに失敗しました: {e}")
            print("Hugging Faceからデータセットをロードします...")
            dataset = prepare_dataset_from_hf(args.dataset_name, tokenizer, hf_tokenizer, 
                                            args.max_seq_len, max_valid_samples=1000, args=args)
    else:
        # Hugging Faceからデータセットを準備
        print("Hugging Faceからデータセットをロードして準備します")
        dataset = prepare_dataset_from_hf(args.dataset_name, tokenizer, hf_tokenizer, 
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
    if hasattr(hf_tokenizer, 'vocab_size'):
        vocab_size = hf_tokenizer.vocab_size
    elif hasattr(hf_tokenizer, 'vocab') and isinstance(hf_tokenizer.vocab, dict):
        vocab_size = len(hf_tokenizer.vocab)
    else:
        # 安全のためのデフォルト値
        vocab_size = 32000
        print(f"警告: トークナイザーから語彙サイズを取得できませんでした。デフォルト値 {vocab_size} を使用します。")
    
    print(f"使用する語彙サイズ: {vocab_size}")
    
    # モデル設定
    print("\nモデルを初期化します...")
    model_config = ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        vocab_size=vocab_size,
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
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mlm_epochs=0,  # MLM学習はスキップ
        diffusion_epochs=args.epochs,  # Diffusion学習のみ実行
        weight_decay=0.01,
        warmup_steps=500,
        use_amp=True,
        use_gradient_checkpointing=True,
        clip_grad_norm=True,
        clip_value=1.0
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
    
    # トークン化 - JapaneseTokenizerとHugging Faceトークナイザーの両方をテスト
    # JapaneseTokenizerにも特殊トークンを追加しないように明示
    if hasattr(tokenizer, 'encode_no_special'):
        jp_tokens_ids = tokenizer.encode_no_special(test_text)
    else:
        jp_tokens_ids = tokenizer.encode(test_text)
    hf_tokens_ids = hf_tokenizer.encode(test_text, add_special_tokens=False)
    
    # デバッグ用のトークン比較
    if len(jp_tokens_ids) != len(hf_tokens_ids):
        print(f"トークン数の不一致: jp={len(jp_tokens_ids)}, hf={len(hf_tokens_ids)}")
        if len(jp_tokens_ids) > len(hf_tokens_ids):
            # 余分なトークンを取り除く（通常はEOSトークン=1）
            jp_tokens_ids = jp_tokens_ids[:-1]
            print(f"JapaneseTokenizerの末尾トークンを取り除きました: {jp_tokens_ids}")
    
    print(f"JapaneseTokenizer トークンID: {jp_tokens_ids}")
    print(f"HuggingFace トークンID: {hf_tokens_ids}")
    print(f"一致しているか: {jp_tokens_ids == hf_tokens_ids}")
    
    # トークン文字列表示
    tokens_str = hf_tokenizer.convert_ids_to_tokens(hf_tokens_ids)
    print(f"トークン: {tokens_str}")
    
    # デコード - 両方のトークナイザーをテスト
    try:
        jp_decoded_text = tokenizer.decode(jp_tokens_ids)
        print(f"JapaneseTokenizer デコード結果: {jp_decoded_text}")
    except Exception as e:
        print(f"JapaneseTokenizer デコードエラー: {e}")
    
    hf_decoded_text = hf_tokenizer.decode(hf_tokens_ids, skip_special_tokens=True)
    print(f"HuggingFace デコード結果: {hf_decoded_text}")
    
    # 特殊トークンの確認
    print(f"\n特殊トークン情報:")
    print(f"  MASK: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
    print(f"  HF MASK: {hf_tokenizer.mask_token} (ID: {hf_tokenizer.mask_token_id})")
    print(f"  語彙サイズ: {hf_tokenizer.vocab_size}")
    
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
            sample_text = hf_tokenizer.decode(sample_ids, skip_special_tokens=True)
            print(f"サンプルテキスト: {sample_text[:100]}..." if len(sample_text) > 100 else sample_text)
    else:
        print("トレーニングデータセットが見つかりません")
    
    # Colab環境の場合、メモリ使用量や各種設定を調整
    if is_colab:
        # バッチサイズの調整
        original_batch_size = training_config.batch_size
        training_config.batch_size = min(training_config.batch_size, 4)
        if original_batch_size != training_config.batch_size:
            print(f"Colab環境向けにバッチサイズを調整: {original_batch_size} → {training_config.batch_size}")
        
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
    tokenizer_for_wrapper = hf_tokenizer  # デフォルトはHuggingFaceトークナイザー
    
    # サンプルテキストでトークナイザーをテスト
    test_text = "これはトークナイザーのテストです。"
    try:
        test_tokens_hf = hf_tokenizer.encode(test_text, add_special_tokens=False)
        print(f"HuggingFaceトークナイザーのテスト結果: {test_tokens_hf[:10]}...")
        # 成功したら、このトークナイザーを使用
        tokenizer_for_wrapper = hf_tokenizer
    except Exception as e1:
        print(f"HuggingFaceトークナイザーのエラー: {e1}")
        try:
            # JapaneseTokenizerでも試す
            test_tokens_jp = tokenizer.encode(test_text)
            print(f"JapaneseTokenizerのテスト結果: {test_tokens_jp[:10]}...")
            # 成功したら、このトークナイザーを使用
            tokenizer_for_wrapper = tokenizer
        except Exception as e2:
            print(f"JapaneseTokenizerのエラー: {e2}")
            print("警告: どちらのトークナイザーも正常に動作しません。")
    
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
        trainer.train_diffusion()
        
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