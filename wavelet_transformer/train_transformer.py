#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ベースラインTransformerモデルの学習スクリプト
比較実験のためのベースラインを構築
"""

import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, set_seed
from transformers import DataCollatorForLanguageModeling

from wavelet_transformer.models.transformer_model import TransformerForMaskedLM
from wavelet_transformer.training.trainer import ModelTrainer
from wavelet_transformer.config import get_transformer_config
from wavelet_transformer.data.dataset import prepare_mlm_dataset
from wavelet_transformer.utils.common import setup_logging, print_model_info

def parse_args():
    parser = argparse.ArgumentParser(description="ベースラインTransformerモデルの学習")
    parser.add_argument("--model_size", type=str, default="base", choices=["small", "base", "large"],
                        help="モデルサイズ: small, base, large")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        choices=["wikitext", "bookcorpus", "oscar", "custom"],
                        help="使用するデータセット")
    parser.add_argument("--dataset_name", type=str, default="wikitext-103-raw-v1",
                        help="データセット名/バージョン")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased",
                        help="使用するトークナイザー")
    parser.add_argument("--output_dir", type=str, default="/content/output",
                        help="出力先ディレクトリ")
    parser.add_argument("--epochs", type=int, default=None,
                        help="エポック数（指定がなければ設定ファイルの値を使用）")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="バッチサイズ（指定がなければ設定ファイルの値を使用）")
    parser.add_argument("--max_seq_len", type=int, default=None,
                        help="最大シーケンス長（指定がなければ設定ファイルの値を使用）")
    parser.add_argument("--debug", action="store_true",
                        help="デバッグモード（少量のデータで実行）")
    parser.add_argument("--seed", type=int, default=42,
                        help="乱数シード")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # シード固定
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 設定取得
    config = get_transformer_config(args.model_size)
    
    # コマンドライン引数による上書き
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_seq_len:
        config.max_seq_length = args.max_seq_len
    if args.output_dir:
        config.output_dir = args.output_dir
        config.log_dir = os.path.join(args.output_dir, "logs")
    
    # デバッグモード設定
    config.debug = args.debug
    
    # 出力ディレクトリの作成
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # ロガー設定
    logger = setup_logging(os.path.join(config.log_dir, "transformer_training.log"))
    logger.info(f"Transformer学習を開始します - サイズ: {args.model_size}")
    logger.info(f"設定: {config}")
    
    # デバイス確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用デバイス: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA バージョン: {torch.version.cuda}")
    
    # トークナイザーのロード
    logger.info(f"トークナイザーをロード中: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # データセットの準備
    logger.info(f"データセットの準備: {args.dataset}/{args.dataset_name}")
    train_dataset, valid_dataset = prepare_mlm_dataset(
        dataset_name=args.dataset,
        dataset_config=args.dataset_name,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        cache_dir=os.path.join(config.output_dir, "cache")
    )
    
    # デバッグモードの場合はデータセットを縮小
    if config.debug:
        train_size = min(config.debug_samples, len(train_dataset))
        valid_size = min(config.debug_samples // 10, len(valid_dataset))
        train_dataset = Subset(train_dataset, range(train_size))
        valid_dataset = Subset(valid_dataset, range(valid_size))
        logger.info(f"デバッグモード: 訓練データ {train_size}サンプル, 検証データ {valid_size}サンプル")
    
    # モデル設定の更新（語彙サイズなど）
    config.vocab_size = len(tokenizer)
    
    # データコレータの準備
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.mlm_probability
    )
    
    # モデルの初期化
    logger.info("Transformerモデルを初期化中...")
    model = TransformerForMaskedLM(config)
    print_model_info(model)
    
    # トレーナーの初期化
    trainer = ModelTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        config=config,
        device=device,
        output_dir=config.output_dir
    )
    
    # 学習実行
    logger.info("学習を開始します...")
    try:
        results = trainer.train(collate_fn=data_collator)
        logger.info(f"学習完了: ベスト検証ロス {results['best_val_loss']:.4f}")
    except Exception as e:
        logger.error(f"学習中にエラーが発生しました: {e}")
        raise
    
    logger.info(f"モデルを保存しました: {os.path.join(config.output_dir, 'checkpoints/best_model.pt')}")
    logger.info("学習プロセスが完了しました")

if __name__ == "__main__":
    main()
