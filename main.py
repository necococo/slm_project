#!/usr/bin/env python
# -*- coding: utf-8 -*-
# main.py
# google colabでの実行を想定しています。
#
# 手順：
#  1) CPUランタイムで  !python main.py
#     => データセットを読み込み・トークナイズし、.save_to_disk(...) して終了
#  2) ランタイムをGPUに切り替え  !python main.py
#     => 前処理済みデータを読み込み、WaveNetworkLMを学習

import os
import sys
import torch
from datetime import datetime
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

from slm.config import ModelConfig, TrainingConfig, PathsConfig
from slm.model import WaveNetworkLM
from slm.trainer import Trainer

def main():
    """
    Wave Network言語モデルの学習を実行するメインスクリプト。
      - CPU環境: データセットの前処理（トークナイズ）だけ行い終了
      - GPU環境: 前処理済みデータを読み込み、学習を実行
    """
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
    paths_config = PathsConfig()
    
    # モデル設定
    model_config = ModelConfig()
    
    # 学習設定
    training_config = TrainingConfig()
    
    # ディレクトリ準備
    os.makedirs(paths_config.data_dir, exist_ok=True)
    os.makedirs(paths_config.checkpoint_dir, exist_ok=True)
    os.makedirs(paths_config.log_dir, exist_ok=True)
    
    # "resume" 引数があれば途中再開
    resume_checkpoint = None
    if len(sys.argv) > 1 and sys.argv[1] == "resume":
        resume_checkpoint = os.path.join(paths_config.checkpoint_dir, "final_model.pt")
        print(f"チェックポイントから復元します: {resume_checkpoint}")
    
    try:
        # CPUモードで前処理
        if device.type == "cpu":
            print("=== CPU環境: データ前処理を行い、終了します ===")
            
            # 1) データ読み込み
            print(f"Loading dataset: {paths_config.dataset_name}/{paths_config.dataset_subset}")
            dataset = load_dataset(paths_config.dataset_name, paths_config.dataset_subset)
            raw_train = dataset["train"]
            
            # 2) train/valid/test に分割（例: 98%/1%/1%）
            print("データセットをtrain/valid/testに分割します...")
            split_data = raw_train.train_test_split(test_size=0.02, seed=42)
            train_dataset = split_data["train"]
            temp_dataset = split_data["test"]
            split_temp = temp_dataset.train_test_split(test_size=0.5, seed=42)
            valid_dataset = split_temp["train"]
            test_dataset = split_temp["test"]
            print(f"Train size: {len(train_dataset)}")
            print(f"Valid size: {len(valid_dataset)}")
            print(f"Test size: {len(test_dataset)}")
            
            # 3) Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(paths_config.tokenizer_name)
            print("Tokenizer vocab_size:", len(tokenizer))
            
            # 4) 前処理 (トークナイズ) 関数
            def tokenize_fn(examples):
                return tokenizer(examples["text"]) # ここでは切り取りや穴埋めは行わず可変長のまま保存します。GPU環境でのtrain直前にdataloaderのcollatorが切り取りや穴埋めをするので長さを調整しながら学習の様子見ができます。
            
            print("Tokenizing dataset ...")
            train_dataset = train_dataset.map(tokenize_fn, batched=True, batch_size=4096, remove_columns=["text"])
            valid_dataset = valid_dataset.map(tokenize_fn, batched=True, batch_size=4096, remove_columns=["text"])
            test_dataset = test_dataset.map(tokenize_fn, batched=True, batch_size=4096, remove_columns=["text"])
            
            # 5) 保存
            print("前処理済みデータをディスクに保存します...")
            train_dataset.save_to_disk(os.path.join(paths_config.data_dir, "train_dataset"))
            valid_dataset.save_to_disk(os.path.join(paths_config.data_dir, "valid_dataset"))
            test_dataset.save_to_disk(os.path.join(paths_config.data_dir, "test_dataset"))
            
            print("=== 前処理完了。ランタイムをGPUに切り替えて再実行してください。 ===")
            sys.exit(0)
        
        # GPUモード: 学習パート
        print("=== GPU環境: 学習を実行します ===")
        
        # 前処理済みデータをロード
        train_dataset = load_from_disk(os.path.join(paths_config.data_dir, "train_dataset"))
        valid_dataset = load_from_disk(os.path.join(paths_config.data_dir, "valid_dataset"))
        
        # Tokenizer読み込み
        tokenizer = AutoTokenizer.from_pretrained(paths_config.tokenizer_name)
        
        # model_configにトークナイザをセット (vocab_sizeなどを取得させるため)
        model_config.set_tokenizer(tokenizer)
        
        # WaveNetworkLM を初期化
        print("WaveNetworkLMを初期化します...")
        model = WaveNetworkLM(model_config)

        # チェックポイント再開
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            from slm.utils import load_checkpoint
            load_checkpoint(resume_checkpoint, model)

        # トレーナー
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            training_config=training_config,
            device=device,
            paths_config=paths_config
        )
        
        # 1) MLM学習
        print("MLM学習を開始...")
        trainer.train_mlm()
        
        # 2) Diffusion Fine-tuning (不要なら0epochにする)
        if training_config.diffusion_epochs > 0:
            print("Diffusionファインチューニングを開始します...")
            trainer.train_diffusion()
        
        # チェックポイント保存
        final_model_path = os.path.join(paths_config.checkpoint_dir, "final_model.pt")
        trainer.save_checkpoint("final_model")
        print(f"モデルを保存しました: {final_model_path}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Trainerリソース解放
        if 'trainer' in locals():
            trainer.close()
        
        print(f"=== 処理完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")


if __name__ == "__main__":
    main()