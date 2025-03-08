#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
英語の小規模データセットでWave Networkを学習し、埋め込み表現の分布を可視化します。
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from slm.config import ModelConfig, TrainingConfig, PathsConfig
from slm.modules.wave_network import WaveNetworkLM
from slm.train import Trainer
from slm.collator import CustomCollator

def parse_arguments():
    """コマンドライン引数を解析します"""
    parser = argparse.ArgumentParser(description='Wave Network トレーニングと埋め込み可視化')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='前処理済みデータセットのディレクトリ（指定がなければ新規ダウンロード）')
    parser.add_argument('--language', type=str, default='en', choices=['en', 'ja'],
                       help='使用する言語（en: 英語, ja: 日本語）')
    parser.add_argument('--dataset', type=str, default=None,
                       help='使用するデータセット名（例："wikitext-2-v1", "wikitext-103-v1"）')
    parser.add_argument('--train_samples', type=int, default=None,
                       help='学習データセットのサンプル数（指定しない場合は全データを使用）')
    parser.add_argument('--valid_samples', type=int, default=None,
                       help='検証データセットのサンプル数（指定しない場合はtrain_samplesの1/10）')
    parser.add_argument('--sample_ratio', type=float, default=None,
                       help='データセットの使用割合 (0.0-1.0)')
    parser.add_argument('--sample_strategy', type=str, default='first', choices=['first', 'random'],
                       help='サンプリング方法 (first: 先頭からサンプル, random: ランダムサンプル)')
    parser.add_argument('--sequence_length', type=int, default=512,
                       help='使用するシーケンス長 (入力トークン数)、長いシーケンスには2048等を指定')
    parser.add_argument('--epochs', type=int, default=2,
                       help='トレーニングエポック数')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='バッチサイズ')
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='隠れ層のサイズ')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='学習率')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='乱数シード値')
    parser.add_argument('--use_cut_cross_entropy', action='store_true', default=True,
                       help='Cut Cross Entropyを使用する（デフォルト: True）')
    parser.add_argument('--no_cut_cross_entropy', action='store_false', dest='use_cut_cross_entropy',
                       help='通常のCross Entropyを使用する')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='学習率ウォームアップのステップ数')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decayの強さ')
    return parser.parse_args()

def setup_environment(args):
    """環境設定とディレクトリ作成を行います"""
    # シード設定
    seed = args.random_seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # パス設定（言語に基づく）
    paths_config = PathsConfig(language=args.language)
    
    # データセットの指定があればオーバーライド
    if args.dataset is not None:
        if args.language == 'en':
            if 'wikitext' in args.dataset:
                paths_config.dataset_name = "wikitext"
                paths_config.dataset_subset = args.dataset
                print(f"データセットを変更: {args.dataset}")
            else:
                print(f"警告: サポートされていないデータセット '{args.dataset}' です。デフォルトを使用します。")
        else:
            print(f"警告: {args.language}言語では現在データセットの変更はサポートされていません。")
    
    # データディレクトリをオーバーライド（指定されている場合）
    if args.data_dir:
        # 指定されたディレクトリの下に "processed" サブディレクトリを作成
        processed_dir = os.path.join(args.data_dir, "processed")
        paths_config.data_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
    
    # ディレクトリ作成
    os.makedirs(paths_config.data_dir, exist_ok=True)
    os.makedirs(paths_config.checkpoint_dir, exist_ok=True)
    os.makedirs(paths_config.log_dir, exist_ok=True)
    os.makedirs(paths_config.visualization_dir, exist_ok=True)
    
    return device, paths_config

def load_dataset_from_disk_or_download(paths_config, args):
    """ディスクから前処理済みデータセットを読み込むか、新規ダウンロードします"""
    if args.data_dir and os.path.exists(args.data_dir):
        print(f"前処理済みデータセットを読み込み中: {args.data_dir}")
        try:
            train_dataset = load_from_disk(os.path.join(args.data_dir, "train"))
            valid_dataset = load_from_disk(os.path.join(args.data_dir, "validation"))
            
            print(f"データセット全体のサイズ - 学習: {len(train_dataset)}件, 検証: {len(valid_dataset)}件")
            
            # データセットの統計情報を表示
            print_dataset_statistics(train_dataset, "学習データ")
            
            # サンプル数調整処理
            train_dataset, valid_dataset = adjust_dataset_size(train_dataset, valid_dataset, args)
            
            return {"train": train_dataset, "validation": valid_dataset}
        except Exception as e:
            print(f"前処理済みデータセットの読み込みエラー: {e}")
            print("新規ダウンロードを試みます...")
    
    # 新規ダウンロード
    dataset = load_english_dataset(paths_config)
    
    # サンプル数調整処理
    dataset["train"], dataset["validation"] = adjust_dataset_size(dataset["train"], dataset["validation"], args)
    
    return dataset

def adjust_dataset_size(train_dataset, valid_dataset, args):
    """データセットのサイズを実行時引数に基づいて調整します"""
    original_train_size = len(train_dataset)
    original_valid_size = len(valid_dataset)
    
    # 訓練データサイズの決定
    train_size = original_train_size
    
    # 明示的なサンプル数が指定された場合
    if args.train_samples is not None:
        train_size = min(args.train_samples, original_train_size)
    # 割合が指定された場合
    elif args.sample_ratio is not None:
        train_size = int(original_train_size * args.sample_ratio)
    
    # 検証データサイズの決定
    valid_size = original_valid_size
    if args.valid_samples is not None:
        valid_size = min(args.valid_samples, original_valid_size)
    elif args.train_samples is not None or args.sample_ratio is not None:
        # 明示的な指定がない場合、訓練データの1/10（最低1サンプル）
        valid_size = max(1, min(int(train_size * 0.1), original_valid_size))
    
    # サイズ調整が必要な場合
    if train_size < original_train_size or valid_size < original_valid_size:
        if args.sample_strategy == 'random':
            # ランダムサンプリング
            train_indices = np.random.choice(original_train_size, train_size, replace=False)
            valid_indices = np.random.choice(original_valid_size, valid_size, replace=False)
            train_dataset = train_dataset.select(train_indices)
            valid_dataset = valid_dataset.select(valid_indices)
            print(f"ランダムサンプリング - 学習: {len(train_dataset)}件, 検証: {len(valid_dataset)}件")
        else:
            # デフォルト: 先頭からサンプリング
            train_dataset = train_dataset.select(range(train_size))
            valid_dataset = valid_dataset.select(range(valid_size))
            print(f"先頭からサンプリング - 学習: {len(train_dataset)}件, 検証: {len(valid_dataset)}件")
    
    return train_dataset, valid_dataset

def load_english_dataset(paths_config):
    """英語データセットをロードします（サイズ調整なし）"""
    print(f"データセットをダウンロード中: {paths_config.dataset_name}/{paths_config.dataset_subset}")
    
    try:
        dataset = load_dataset(paths_config.dataset_name, paths_config.dataset_subset)
        print(f"データセットをロードしました - 学習: {len(dataset['train'])}件, 検証: {len(dataset['validation'])}件")
        
        # データセットの統計情報を表示
        print_dataset_statistics(dataset["train"], "学習データ")
        
        return dataset
    except Exception as e:
        print(f"データセットのダウンロードエラー: {e}")
        raise

def print_dataset_statistics(dataset, name=""):
    """データセットのトークン長統計情報を表示します"""
    try:
        # 入力IDsの長さを計算（すでにトークン化されている場合とテキストのみの場合を処理）
        if "input_ids" in dataset[0]:
            lengths = [len(example["input_ids"]) for example in dataset]
        elif "text" in dataset[0]:
            # テキストの長さを概算（粗い近似）
            lengths = [len(example["text"].split()) for example in dataset]
        else:
            print(f"{name}の統計情報を計算できません: 不明なデータ形式")
            return
            
        if not lengths:
            print(f"{name}の統計情報を計算できません: 空のデータセット")
            return
            
        import numpy as np
        min_len = np.min(lengths)
        max_len = np.max(lengths)
        mean_len = np.mean(lengths)
        median_len = np.median(lengths)
        p90 = np.percentile(lengths, 90)
        p95 = np.percentile(lengths, 95)
        p99 = np.percentile(lengths, 99)
        
        print(f"\n{name}トークン長の統計:")
        print(f"  - 最小: {min_len:.1f} トークン")
        print(f"  - 最大: {max_len:.1f} トークン")
        print(f"  - 平均: {mean_len:.1f} トークン")
        print(f"  - メディアン: {median_len:.1f} トークン")
        print(f"  - 90パーセンタイル: {p90:.1f} トークン")
        print(f"  - 95パーセンタイル: {p95:.1f} トークン")
        print(f"  - 99パーセンタイル: {p99:.1f} トークン")
    except Exception as e:
        print(f"{name}の統計情報計算中にエラーが発生しました: {e}")

def setup_tokenizer_and_model(paths_config, args):
    """トークナイザーとモデルをセットアップします"""
    # トークナイザーの読み込み
    print(f"Loading tokenizer: {paths_config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(paths_config.tokenizer_name)
    
    # GPT-2トークナイザーの場合、パディングトークンが設定されていないため追加
    if paths_config.tokenizer_name == "gpt2" and tokenizer.pad_token is None:
        print("GPT-2トークナイザーにパディングトークンを設定します")
        tokenizer.pad_token = tokenizer.eos_token
    
    # モデル設定
    model_config = ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=1,     # レイヤー数も少なく
        vocab_size=None,  # トークナイザーから自動取得
        max_seq_len=args.sequence_length,
        dropout_prob=0.1,
        use_rope=True,
        use_wavelet=False,  # 波変換は今回無効に
        use_cut_cross_entropy=getattr(args, 'use_cut_cross_entropy', True)  # 引数から取得、なければTrue
    )
    model_config.set_tokenizer(tokenizer)
    
    # モデル初期化
    model = WaveNetworkLM(model_config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"シーケンス長: {args.sequence_length}, 隠れ層サイズ: {args.hidden_size}")
    print(f"使用する損失関数: {'Cut Cross Entropy' if model.use_cut_cross_entropy else '通常のCross Entropy'}")
    
    return tokenizer, model, model_config

def prepare_data_for_training(dataset, tokenizer, model_config, batch_size=8):
    """学習用データを準備します"""
    # 入力がすでに辞書型かどうかをチェック
    if isinstance(dataset, dict):
        # すでに前処理されているデータセット（辞書型）を使用
        print("前処理済みデータセットを使用します")
        train_dataset = dataset['train']
        valid_dataset = dataset['validation']
        
        # シーケンス長の調整はせず、コレーターに任せる
        print(f"コレーター内でシーケンス長 {model_config.max_seq_len} に動的調整します")
        
    else:
        # テキストデータの前処理が必要
        print("データセットをトークン化しています...")
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                truncation=True,
                max_length=None,  # パディングせず、完全なシーケンスを保存
                padding=False      # パディングなし
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=["text"]
        )
        
        train_dataset = tokenized_dataset["train"]
        valid_dataset = tokenized_dataset["validation"]
    
    # データコレーターの作成（ここでシーケンス長が適用される）
    collator = CustomCollator(
        tokenizer=tokenizer,
        model_config=model_config,  # max_seq_lenを含む
        mlm=True,
        mlm_probability=0.15,
        mask_token_id=tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else None,
        dynamic_padding=True  # 動的パディングを有効化
    )
    
    # データローダーの作成
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=collator,  # カスタムコレーターが指定シーケンス長でパディング
        shuffle=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        collate_fn=collator,
        shuffle=False
    )
    
    return train_loader, valid_loader, train_dataset, valid_dataset

def train_model(model, train_dataset, valid_dataset, model_config, device, paths_config):
    """モデルを学習します"""
    # 学習設定
    training_config = TrainingConfig(
        learning_rate=1e-3,  # 高めの学習率を維持
        batch_size=8,        # 小さいバッチサイズ
        mlm_epochs=2,        # 少ないエポック数
        mlm_probability=0.15,
        weight_decay=0.01,
        warmup_steps=100,
        use_amp=True,
        clip_grad_norm=True,
        clip_value=1.0
    )
    
    # トレーナー初期化
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        training_config=training_config,
        device=device,
        paths_config=paths_config
    )
    
    print("\nStarting MLM training...")
    # MLM学習の実行
    trainer.train_mlm()
    
    # モデル保存
    trainer.save_checkpoint("english_model")
    checkpoint_path = os.path.join(paths_config.checkpoint_dir, "english_model.pt")
    print(f"Model saved to {checkpoint_path}")
    
    return model

def extract_embeddings(model, dataset, tokenizer, model_config, device, num_samples=100):
    """モデルから埋め込み表現を抽出します"""
    model.eval()
    
    # 抽出する埋め込み
    sentence_real_embeds = []
    sentence_imag_embeds = []
    token_real_embeds = []
    token_imag_embeds = []
    
    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Extracting embeddings"):
            # 安全なインデックスアクセス
            idx = PathsConfig.safe_index(i)
            example = dataset[idx]
            
            # 入力の準備
            input_ids = torch.tensor([example["input_ids"][:model_config.max_seq_len]]).to(device)
            attention_mask = torch.tensor([example["attention_mask"][:model_config.max_seq_len]]).to(device)
            
            # まずトークン埋め込みを取得
            token_embedding = model.token_embedding(input_ids)
            
            # 波表現に変換 (モジュールからcompute_wave_representation関数をインポート)
            from slm.modules.wave_network import compute_wave_representation
            
            # トークンレベル (ローカル) 波表現を計算
            real_part_token, imag_part_token = compute_wave_representation(token_embedding, global_mode=False)
            
            # 文章レベル (グローバル) 波表現を計算
            real_part_sent, imag_part_sent = compute_wave_representation(token_embedding, global_mode=True)
            
            # センテンスレベル埋め込み（シーケンスの平均）
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(real_part_token)
            sentence_real = (real_part_sent * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            sentence_imag = (imag_part_sent * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            
            # 結果を収集
            sentence_real_embeds.append(sentence_real.cpu().numpy())
            sentence_imag_embeds.append(sentence_imag.cpu().numpy())
            token_real_embeds.append(real_part_token[0].cpu().numpy())  # バッチ内の最初の例
            token_imag_embeds.append(imag_part_token[0].cpu().numpy())  # バッチ内の最初の例
    
    # 結果を連結
    sentence_real_embeds = np.concatenate(sentence_real_embeds, axis=0)
    sentence_imag_embeds = np.concatenate(sentence_imag_embeds, axis=0)
    token_real_embeds = np.concatenate(token_real_embeds, axis=0)
    token_imag_embeds = np.concatenate(token_imag_embeds, axis=0)
    
    # 統計情報をログ出力して確認
    print("\n波表現の統計情報:")
    print(f"トークンレベル実部 - 平均値: {np.mean(token_real_embeds):.6f}, 標準偏差: {np.std(token_real_embeds):.6f}")
    print(f"トークンレベル虚部 - 平均値: {np.mean(token_imag_embeds):.6f}, 標準偏差: {np.std(token_imag_embeds):.6f}")
    print(f"センテンスレベル実部 - 平均値: {np.mean(sentence_real_embeds):.6f}, 標準偏差: {np.std(sentence_real_embeds):.6f}")
    print(f"センテンスレベル虚部 - 平均値: {np.mean(sentence_imag_embeds):.6f}, 標準偏差: {np.std(sentence_imag_embeds):.6f}")
    
    return {
        'sentence_real': sentence_real_embeds,
        'sentence_imag': sentence_imag_embeds,
        'token_real': token_real_embeds,
        'token_imag': token_imag_embeds
    }

def visualize_embeddings(embeddings, paths_config):
    """埋め込み表現の分布を可視化します"""
    plt.figure(figsize=(20, 15))
    
    # センテンスレベル実部
    plt.subplot(2, 2, 1)
    plt.hist(embeddings['sentence_real'].flatten(), bins=100, alpha=0.7)
    plt.title('Sentence Level - Real Part Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # センテンスレベル虚部
    plt.subplot(2, 2, 2)
    plt.hist(embeddings['sentence_imag'].flatten(), bins=100, alpha=0.7)
    plt.title('Sentence Level - Imaginary Part Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # トークンレベル実部
    plt.subplot(2, 2, 3)
    plt.hist(embeddings['token_real'].flatten(), bins=100, alpha=0.7)
    plt.title('Token Level - Real Part Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # トークンレベル虚部
    plt.subplot(2, 2, 4)
    plt.hist(embeddings['token_imag'].flatten(), bins=100, alpha=0.7)
    plt.title('Token Level - Imaginary Part Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(paths_config.visualization_path, "embedding_distributions.png")
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    
    # 追加の統計情報
    stats = {}
    for key, value in embeddings.items():
        flat_values = value.flatten()
        stats[key] = {
            'mean': np.mean(flat_values),
            'std': np.std(flat_values),
            'max': np.max(flat_values),
            'min': np.min(flat_values),
            'abs_mean': np.mean(np.abs(flat_values))
        }
    
    # 統計情報の保存
    stats_path = os.path.join(paths_config.visualization_path, "embedding_stats.txt")
    with open(stats_path, 'w') as f:
        f.write("Embedding Distribution Statistics\n")
        f.write("===============================\n\n")
        for key, stat in stats.items():
            f.write(f"{key}:\n")
            for stat_name, stat_value in stat.items():
                f.write(f"  {stat_name}: {stat_value}\n")
            f.write("\n")
    
    print(f"Statistics saved to {stats_path}")
    
    # 追加の可視化：複素平面上の散布図（サンプル）
    plt.figure(figsize=(15, 15))
    
    # センテンスレベル（最初の50要素）
    plt.subplot(1, 2, 1)
    plt.scatter(
        embeddings['sentence_real'][0, :50],
        embeddings['sentence_imag'][0, :50],
        alpha=0.7
    )
    plt.title('Sentence Level - Complex Plane (First 50 dims)')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # トークンレベル（最初のトークン、最初の50要素）
    plt.subplot(1, 2, 2)
    plt.scatter(
        embeddings['token_real'][0, :50],
        embeddings['token_imag'][0, :50],
        alpha=0.7
    )
    plt.title('Token Level - Complex Plane (First 50 dims)')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    complex_save_path = os.path.join(paths_config.visualization_path, "complex_plane.png")
    plt.savefig(complex_save_path)
    print(f"Complex plane visualization saved to {complex_save_path}")

def main():
    """メイン実行関数"""
    # 引数解析
    args = parse_arguments()
    
    print(f"=== Wave Network 学習・埋め込み分析 ({args.language}) ===")
    
    # 環境設定
    device, paths_config = setup_environment(args)
    
    # データセットのロード（前処理済みorダウンロード）
    try:
        dataset = load_dataset_from_disk_or_download(paths_config, args)
        
        # トークナイザーとモデルのセットアップ（シーケンス長指定）
        tokenizer, model, model_config = setup_tokenizer_and_model(paths_config, args)
        model.to(device)
        
        # データセットのタイプに関する情報を表示
        print(f"選択されたデータセット: {paths_config.dataset_name}/{paths_config.dataset_subset}")
        if "wikitext-103" in paths_config.dataset_subset:
            print("注意: WikiText-103は大規模データセットで、より長いシーケンスを含んでいます")
            print(f"シーケンス長が{model_config.max_seq_len}に設定されています。長いシーケンスを活用するには2048程度が推奨されます。")
        
        # データ準備
        train_loader, valid_loader, train_dataset, valid_dataset = prepare_data_for_training(
            dataset, tokenizer, model_config, batch_size=args.batch_size
        )
        
        # モデル学習
        print("\nモデルの学習を開始します...")
        
        # 学習設定をコマンドライン引数で上書き
        training_config = TrainingConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            mlm_epochs=args.epochs,
            mlm_probability=0.15,
            weight_decay=getattr(args, 'weight_decay', 0.01),
            warmup_steps=getattr(args, 'warmup_steps', 100),
            use_amp=True,
            clip_grad_norm=True,
            clip_value=1.0  # 勾配爆発を防ぐ適切な値を設定
        )
        
        # トレーナー初期化とトレーニング
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            training_config=training_config,
            device=device,
            paths_config=paths_config
        )
        
        trainer.train_mlm()
        
        print("\n埋め込み抽出を行います...")
        # 埋め込み抽出と可視化
        embeddings = extract_embeddings(model, valid_dataset, tokenizer, model_config, device)
        print("\n埋め込み分布を可視化しています...")
        visualize_embeddings(embeddings, paths_config)
        
        print("\n分析完了!")
    except Exception as e:
        print(f"実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()