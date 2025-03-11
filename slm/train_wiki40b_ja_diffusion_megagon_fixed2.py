#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from slm.config import ModelConfig, TrainingConfig, PathsConfig
from slm.modules.wave_network import WaveNetworkLM
from slm.wiki40b_ja_dataset import WikiDataset, collate_fn
from slm.tokenizer import JapaneseTokenizer
from slm.train import Trainer

# データセットラッパークラスを追加
class HFDatasetWrapper(Dataset):
    """Hugging Faceデータセットをラップしてcollatorに適した形式を提供するクラス"""
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # collatorが期待する形式にデータを変換
        if isinstance(item, dict):
            # input_idsが直接存在するか確認
            if "input_ids" in item:
                return {"input_ids": item["input_ids"], 
                        "attention_mask": item.get("attention_mask", [1] * len(item["input_ids"]))}
            else:
                # データセットの形式をログ出力して確認
                print(f"Warning: Dataset item at index {idx} does not have 'input_ids': {item.keys()}")
                
                # データセットの形式に基づいて適切に変換
                # 例えば、データセットが異なる構造を持っている場合
                if "tokens" in item:  # トークンIDがtokensキーに格納されている場合
                    return {"input_ids": item["tokens"], 
                            "attention_mask": item.get("mask", [1] * len(item["tokens"]))}
                else:
                    # 何らかのキーを使って強制的に変換
                    first_key = list(item.keys())[0]
                    if isinstance(item[first_key], list):
                        # リスト形式のデータであれば、それをinput_idsとして使用
                        return {"input_ids": item[first_key], 
                                "attention_mask": [1] * len(item[first_key])}
                    else:
                        # そうでなければ、空のデータを返す（エラー回避）
                        return {"input_ids": [], "attention_mask": []}
        
        # リスト形式の場合（例えばトークンIDのリスト）
        elif isinstance(item, list):
            return {"input_ids": item, "attention_mask": [1] * len(item)}
        
        # その他の場合は空の辞書を返す（エラー回避）
        return {"input_ids": [], "attention_mask": []}

def parse_args():
    parser = argparse.ArgumentParser(description="toramaru-u/wiki40b-ja データセットを使用したDiffusionモデルの学習")
    
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
    
    return parser.parse_args()

def prepare_dataset_from_hf(dataset_name, tokenizer, hf_tokenizer, max_seq_len, max_valid_samples=1000):
    """Hugging Faceからデータセットをロードして準備する"""
    print(f"Hugging Faceからデータセット {dataset_name} をロード中...")
    dataset = load_dataset(dataset_name)
    
    print(f"データセット情報:")
    for split in dataset:
        print(f"  {split}: {len(dataset[split])}サンプル")
    
    # トークン化関数
    def tokenize_function(examples):
        tokenized = {"input_ids": [], "attention_mask": []}
        
        for text in examples["text"]:
            # トークン化 - 直接Hugging Faceトークナイザーを使用
            token_ids = hf_tokenizer.encode(text, add_special_tokens=False)
            
            # 最大長に切り詰め
            if len(token_ids) > max_seq_len:
                token_ids = token_ids[:max_seq_len]
            
            # 注意マスクを作成（すべて1）
            attn_mask = [1] * len(token_ids)
            
            tokenized["input_ids"].append(token_ids)
            tokenized["attention_mask"].append(attn_mask)
        
        return tokenized
    
    # データセットをトークン化
    print("データセットをトークン化中...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=["text"]
    )
    
    # 検証セットのサイズを制限（メモリ節約のため）
    if "validation" in tokenized_datasets and max_valid_samples is not None:
        tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(min(len(tokenized_datasets["validation"]), max_valid_samples)))
    
    print("トークン化済みデータセット情報:")
    for split in tokenized_datasets:
        print(f"  {split}: {len(tokenized_datasets[split])}サンプル")
    
    # データセットの形式を確認
    if "train" in tokenized_datasets and len(tokenized_datasets["train"]) > 0:
        first_item = tokenized_datasets["train"][0]
        print(f"最初のアイテムのキー: {list(first_item.keys())}")
        if "input_ids" in first_item:
            print(f"input_idsの長さ: {len(first_item['input_ids'])}")
        else:
            print(f"警告: 'input_ids'キーがデータセットのアイテムに見つかりません")
    
    return tokenized_datasets

def load_tokenizer_megagon(tokenizer_name):
    """megagonlabs/t5-base-japanese-webなどのトークナイザーをロード"""
    print(f"トークナイザー {tokenizer_name} をロード中...")
    
    # まずHuggingFaceからロード
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
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
    except ImportError:
        is_colab = False
        print("ローカル環境で実行中")
    
    # トークナイザーのロード
    tokenizer, hf_tokenizer = load_tokenizer_megagon(args.tokenizer_name)
    mask_token_id = tokenizer.mask_token_id
    
    print(f"[MASK]トークンID: {mask_token_id}")
    
    # データセットの準備
    if args.use_local_dataset:
        print(f"ローカルデータセットを使用: {args.local_data_dir}")
        # ローカルデータセットの読み込み
        from datasets import load_from_disk
        try:
            dataset = load_from_disk(args.local_data_dir)
            print("データセットを正常にロードしました")
        except Exception as e:
            print(f"ローカルデータセットのロードに失敗しました: {e}")
            print("Hugging Faceからデータセットをロードします...")
            dataset = prepare_dataset_from_hf(args.dataset_name, tokenizer, hf_tokenizer, args.max_seq_len)
    else:
        # Hugging Faceからデータセットを準備
        dataset = prepare_dataset_from_hf(args.dataset_name, tokenizer, hf_tokenizer, args.max_seq_len)
        
        # 将来の使用のためにローカルに保存
        if not is_colab:  # Colab環境ではディスク容量節約のため保存しない
            os.makedirs(args.local_data_dir, exist_ok=True)
            print(f"データセットをローカルに保存: {args.local_data_dir}")
            dataset.save_to_disk(args.local_data_dir)
    
    # モデル設定
    model_config = ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        vocab_size=hf_tokenizer.vocab_size,
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
        run_name=f"wiki40b_ja_diffusion_megagon_{args.hidden_size}h_{args.num_layers}l"
    )
    
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
    jp_tokens_ids = tokenizer.encode(test_text)
    hf_tokens_ids = hf_tokenizer.encode(test_text, add_special_tokens=False)
    
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
    
    # データセットをラッパーでラップしてcollatorに適した形式に変換
    train_dataset = HFDatasetWrapper(dataset["train"]) if "train" in dataset else None
    valid_dataset = HFDatasetWrapper(dataset["validation"]) if "validation" in dataset else None
    
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
        
        # Diffusion訓練の実行
        print("Diffusionモデルの学習を開始します...")
        trainer.train_diffusion()
        
        # 最終チェックポイントの保存
        trainer.save_checkpoint("final_model")
        
        # リソースの解放
        trainer.close()
        
        print("学習が完了しました。最終モデルが保存されました。")
    except Exception as e:
        print(f"トレーニング中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()