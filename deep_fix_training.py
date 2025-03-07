#!/usr/bin/env python
# -*- coding: utf-8 -*-
# main.py
# google colabでの実行を想定しています。

# ①初回はランタイムをCPU環境で前処理を行い、データセットがディスクに保存され一旦終わるようになっています。ランタイムをGPU環境に切り替えてから再度 !python main.py を実行してください。データセットがdiskから読み込まれます。
# ②学習の再開を行いたい場合は、コマンドライン引数 "resume" を渡して実行する（!python main.py resume）と、保存済みのチェックポイントから復元して学習を続行できます。

import os
import sys
import torch
from datetime import datetime

from slm.config import ModelConfig, TrainingConfig, PathsConfig
from slm.model import WaveNetworkLM
from slm.trainer import Trainer
from slm.data_loader import get_dataset
from slm.tokenizer import JapaneseTokenizer

def main():
    """
    How:
        Wave Network言語モデルの学習を実行します。
        1. 設定の初期化
        2. データセットとトークナイザーの準備
        3. モデルの初期化
        4. トレーナーでMLM学習とDiffusion Fine-tuningを実行
    """
    # 開始メッセージ
    print(f"=== Wave Network言語モデル学習 開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    # シード設定
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # パス設定
    paths_config = PathsConfig(
        base_dir="/content/drive/MyDrive/slm",
        dataset_name="singletongue/wikipedia-utils",  # 日本語Wikipediaデータ
        dataset_subset="corpus-jawiki-20230403-filtered-large"  # 使用するサブセットに変更
    )
    
    # モデル設定
    model_config = ModelConfig(
    )
    
    # 学習設定
    training_config = TrainingConfig(
    )
    
    # ディレクトリ作成
    os.makedirs(paths_config.data_dir, exist_ok=True)
    os.makedirs(paths_config.checkpoint_dir, exist_ok=True)
    os.makedirs(paths_config.log_dir, exist_ok=True)
    
    # もし'resume'引数が指定されていたら、保存されたチェックポイントから復元する
    resume_checkpoint = None
    if len(sys.argv) > 1 and sys.argv[1] == "resume":
        resume_checkpoint = os.path.join(paths_config.checkpoint_dir, "final_model.pt")
        print(f"チェックポイントから復元します: {resume_checkpoint}")
    
    try:
        # GPU環境なら、前処理済みのtokenizerとdatasetを読み込みます
        if device.type == "cuda":
            print("GPU環境と判断しました。保存済みのtokenizerとdatasetをロードします。")
            if os.path.exists(paths_config.tokenizer_path):
                tokenizer = JapaneseTokenizer(model_file=paths_config.tokenizer_path)
            else:
                raise ValueError("GPU環境ではtokenizerが保存済みである必要があります。")
            from datasets import load_from_disk
            train_dataset = load_from_disk(os.path.join(paths_config.data_dir, "train_dataset"))
            if os.path.exists(os.path.join(paths_config.data_dir, "valid_dataset")):
                valid_dataset = load_from_disk(os.path.join(paths_config.data_dir, "valid_dataset"))
            else:
                valid_dataset = None

                    # 学習用サブセット
            train_subset = train_dataset.select(range(min(5000, len(train_dataset))))
            valid_subset = valid_dataset.select(range(min(50, len(valid_dataset))))

        else:
            # CPU環境の場合（前処理中）
            print("トークナイザーを準備中...")
            if os.path.exists(paths_config.tokenizer_path):
                tokenizer = JapaneseTokenizer(model_file=paths_config.tokenizer_path)
            else:
                tokenizer = JapaneseTokenizer(
                    hf_model=paths_config.tokenizer_name, 
                    save_to=paths_config.tokenizer_path, 
                    model_file=None
                )
            print(f"語彙サイズ: {tokenizer.vocab_size}")
            
            print("データセットを準備中...")
            train_dataset, valid_dataset = get_dataset(tokenizer, paths_config, max_seq_len=model_config.max_seq_len)
            print(f"学習データサイズ: {len(train_dataset)}件")
            if valid_dataset:
                print(f"検証データサイズ: {len(valid_dataset)}件")
            
            print("前処理済みデータセットをディスクに保存中...")
            train_dataset.save_to_disk(os.path.join(paths_config.data_dir, "train_dataset"))
            if valid_dataset:
                valid_dataset.save_to_disk(os.path.join(paths_config.data_dir, "valid_dataset"))
            print("保存完了。")
            
            print("CPUランタイムです。GPU環境で再実行してください。")
            exit(0)
        
        # ここで model_config に tokenizer をセットする
        model_config.set_tokenizer(tokenizer)
        
        # モデル初期化
        print("モデルを初期化中...")
        model = WaveNetworkLM(model_config)
        
        # もしチェックポイントが存在すれば復元
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            from slm.utils import load_checkpoint
            load_checkpoint(resume_checkpoint, model)  # optimizer等も必要なら適宜復元してください
        
        # トレーナー初期化
        trainer = Trainer(
            model=model,
            train_dataset=train_subset,
            valid_dataset=valid_subset,
            training_config=training_config,
            device=device,
            paths_config=paths_config
        )
        
        # MLM学習
        print("MLM学習を開始します...")
        trainer.train_mlm()
        
        # 最終チェックポイント保存
        final_model_path = os.path.join(paths_config.checkpoint_dir, "final_model.pt")
        trainer.save_checkpoint("final_model")
        print(f"モデルを保存しました: {final_model_path}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # リソース解放
        if 'trainer' in locals():
            trainer.close()
        
        print(f"=== 処理完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")


if __name__ == "__main__":
    main()