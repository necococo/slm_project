#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比較用の標準的なTransformerモデルをトレーニングして
Wave Networkの性能を相対的に評価するためのスクリプト
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoConfig,
    BertConfig,
    BertForMaskedLM,  # GPT2からBERTに変更（MLMタスクに適合）
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from slm.config import PathsConfig

def main():
    print(f"=== Transformer Baseline学習 開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # シード固定
    seed = 3407
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # パス設定
    paths_config = PathsConfig()
    
    # ディレクトリ準備
    baseline_dir = os.path.join(paths_config.checkpoint_dir, "baseline_model")
    os.makedirs(baseline_dir, exist_ok=True)
    
    try:
        if device.type != "cuda":
            print("このスクリプトはGPU環境で実行してください")
            sys.exit(1)
            
        # データセット読み込み（学習時間短縮のため小さなサブセットを使用）
        train_dataset = load_from_disk(os.path.join(paths_config.data_dir, "train_dataset"))
        valid_dataset = load_from_disk(os.path.join(paths_config.data_dir, "valid_dataset"))
        
        # サブセット作成
        train_subset = train_dataset.select(range(min(100000, len(train_dataset))))
        valid_subset = valid_dataset.select(range(min(5000, len(valid_dataset))))
        
        print(f"訓練データサイズ: {len(train_subset)}")
        print(f"検証データサイズ: {len(valid_subset)}")
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(paths_config.tokenizer_name)
        
        # モデル設定 (Wave Networkの極小モデルと同程度のサイズ)
        config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=512
        )
        
        # モデル初期化
        model = BertForMaskedLM(config)
        print(f"パラメータ数: {sum(p.numel() for p in model.parameters())/1000000:.2f}M")
        
        # 学習設定 (wandbを無効化)
        training_args = TrainingArguments(
            output_dir=baseline_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,  # 保存するチェックポイント数を制限
            logging_dir=os.path.join(paths_config.log_dir, "baseline"),
            logging_steps=100,
            learning_rate=5e-5,
            weight_decay=0.01,
            push_to_hub=False,
            report_to=[],  # wandbを含むすべてのレポーティングを無効化
            run_name="bert_baseline",  # 一意の実行名
            dataloader_num_workers=4,  # データローダーの並列処理を有効化
            fp16=True  # 混合精度学習を有効化
        )
        
        # データコレータ
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
        
        # トークナイザーのパディングトークンIDの設定を確認
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # トレーナー
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_subset,
            eval_dataset=valid_subset
        )
        
        # 学習実行
        print("BERTベースラインモデル学習開始...")
        trainer.train()
        
        # 評価
        eval_results = trainer.evaluate()
        print(f"評価結果: {eval_results}")
        
        # モデル保存
        trainer.save_model(os.path.join(baseline_dir, "final"))
        print(f"モデル保存完了: {os.path.join(baseline_dir, 'final')}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"=== 処理完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

if __name__ == "__main__":
    main()
