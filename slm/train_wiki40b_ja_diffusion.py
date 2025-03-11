#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import torch
from torch.utils.data import DataLoader
from slm.config import ModelConfig, TrainingConfig, PathsConfig
from slm.modules.wave_network import WaveNetworkLM
from slm.wiki40b_ja_dataset import WikiDataset, load_tokenizer, collate_fn
from slm.train import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Wiki40B日本語データセットを使用したDiffusionモデルの学習")
    
    # データパス関連
    parser.add_argument("--data_dir", type=str, default="/content/drive/MyDrive/slm/data/wiki40b_ja",
                        help="Wiki40B日本語データセットとトークナイザーのディレクトリ")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/slm/outputs",
                        help="モデル出力ディレクトリ")
    parser.add_argument("--model_prefix", type=str, default="sp_jwiki",
                        help="SentencePieceモデルのプレフィックス")
    
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

def main():
    args = parse_args()
    
    # 乱数シード設定
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # トークナイザーのロード
    tokenizer = load_tokenizer(args.data_dir, args.model_prefix)
    mask_token_id = tokenizer.piece_to_id("[MASK]")
    
    print(f"トークナイザーをロードしました。語彙サイズ: {tokenizer.get_piece_size()}")
    print(f"[MASK]トークンID: {mask_token_id}")
    
    # Google Colab環境の検出
    is_colab = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False
    
    # Colab環境での最適化設定
    memory_efficient = True if is_colab else False
    max_samples = None  # 全データを使用する場合はNone
    
    # データセットがGoogleドライブにある場合は警告
    if is_colab and "/content/drive/" in args.data_dir:
        print("警告: GoogleドライブからのI/Oはパフォーマンスに影響する可能性があります。")
        print("       可能であれば、データを/contentにコピーすることを検討してください。")
    
    print(f"データセット設定: memory_efficient={memory_efficient}, max_samples={max_samples}")
    
    # データセットの準備
    # トレーニングデータセット
    train_file_path = os.path.join(args.data_dir, "wiki40b_ja_train.txt")
    train_dataset = WikiDataset(
        file_path=train_file_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        memory_efficient=memory_efficient,
        max_samples=max_samples
    )
    
    # 検証データセット
    valid_file_path = os.path.join(args.data_dir, "wiki40b_ja_valid.txt")
    valid_dataset = WikiDataset(
        file_path=valid_file_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        memory_efficient=memory_efficient,
        max_samples=1000  # 検証セットは小さめに設定
    )
    
    print(f"データセットを準備しました。トレーニングサンプル数: {len(train_dataset)}, 検証サンプル数: {len(valid_dataset)}")
    
    # モデル設定
    model_config = ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        vocab_size=tokenizer.get_piece_size(),
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
        run_name=f"wiki40b_ja_diffusion_{args.hidden_size}h_{args.num_layers}l"
    )
    
    # モデルのインスタンス化
    model = WaveNetworkLM(model_config)
    
    # モデル情報の出力
    param_count = sum(p.numel() for p in model.parameters())
    print(f"モデルを初期化しました。パラメータ数: {param_count:,}")
    
    # Colab環境の場合、メモリ使用量や各種設定を調整
    if is_colab:
        # バッチサイズの調整
        original_batch_size = training_config.batch_size
        training_config.batch_size = min(training_config.batch_size, 4)
        if original_batch_size != training_config.batch_size:
            print(f"Colab環境向けにバッチサイズを調整: {original_batch_size} → {training_config.batch_size}")
        
        # AMP（Automatic Mixed Precision）の確認
        # Colab環境ではTPU用の設定は使用しない
        if hasattr(training_config, 'use_amp') and training_config.use_amp:
            import torch.cuda
            if not torch.cuda.is_available():
                print("GPUが利用できないため、AMPを無効にします")
                training_config.use_amp = False
        
        print("Google Colab環境向けに設定を最適化しました")
    
    # トレーナーの初期化
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

if __name__ == "__main__":
    main()