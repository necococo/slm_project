# train_and_analyze.py
"""Wave Network モデルの学習と埋め込み分析を行うスクリプト"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from slm.config import PathsConfig
from slm.config import ModelConfig
from slm.config import TrainingConfig
from slm.train import Trainer
# 適切なインポートに修正
from slm.modules.wave_network import WaveNetworkLM
# 分析機能を専用モジュールからインポート
from slm.tools.analysis import extract_embeddings, visualize_embeddings

def setup_environment(config):
    """環境のセットアップを行う"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 実行名が指定されているか確認
    run_name = getattr(config, "run_name", None)
    
    # PathsConfigをconfigの各項目から初期化
    # 必須パラメータの確認
    if not hasattr(config, "dataset_name"):
        raise ValueError("dataset_nameはconfigで必須です")
    
    # PathsConfigへ渡す引数を辞書として構築
    paths_args = {
        "dataset_name": config.dataset_name,
        "run_name": run_name
    }
    
    # オプションのパラメータを設定
    optional_params = {
        "base_dir": os.getcwd(),
        "dataset_subset": None,
        "tokenizer_name": "cl-tohoku/bert-base-japanese-whole-word-masking",
        "output_dir": None,
        "cache_dir": os.path.join(os.getcwd(), "cache")
    }
    
    # configに存在するパラメータを優先して使用
    for param, default in optional_params.items():
        paths_args[param] = getattr(config, param, default)
    
    # PathsConfigを初期化
    paths_config = PathsConfig(**paths_args)
    # 正しいディレクトリ群を作成
    os.makedirs(paths_config.checkpoint_dir, exist_ok=True)
    os.makedirs(paths_config.visualization_path, exist_ok=True)
    os.makedirs(os.path.join(paths_config.checkpoint_dir, "tokenizers"), exist_ok=True)
    os.makedirs(paths_config.logs_path, exist_ok=True)
    
    return device, paths_config


def load_dataset_from_disk_or_download(paths_config: PathsConfig, config):
    """ディスクからデータセットをロードするか、ない場合はダウンロードする"""
    try:
        # データセットのロード
        dataset_name = paths_config.dataset_name
        dataset_subset = paths_config.dataset_subset
        print(f"データセット {dataset_name}/{dataset_subset} をロードします...")
        dataset = load_dataset(
        dataset_name,
        dataset_subset,
        cache_dir=paths_config.cache_dir,
        trust_remote_code=True
        )

        print("データセットのロードが完了しました！")
        return dataset
    except Exception as e:
        print(f"データセットのロード中にエラーが発生しました: {e}")
        raise


def setup_tokenizer_and_model(paths_config: PathsConfig, config):
    """トークナイザーとモデルをセットアップする"""
    # トークナイザーのロード
    tokenizer_name = getattr(config, 'model_name', 'cl-tohoku/bert-base-japanese-whole-word-masking')
    print(f"トークナイザー {tokenizer_name} をロードしています...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=paths_config.cache_dir, use_fast=False)
    
    # モデル設定
    model_config = ModelConfig(
        hidden_size=getattr(config, 'hidden_size', 768),
        num_layers=getattr(config, 'num_layers', 3),
        max_seq_len=getattr(config, 'max_seq_len', 512),
        dropout_prob=getattr(config, 'dropout_prob', 0.1),
        use_rope=getattr(config, 'use_rope', True)
    )
    
    # トークナイザーを設定してvocab_sizeを自動取得できるようにする
    model_config.set_tokenizer(tokenizer)
    
    # Wave Networkモデルの初期化
    print(f"Wave Networkモデルを初期化しています...")
    model = WaveNetworkLM(model_config)
    
    return tokenizer, model, model_config


def prepare_data_for_training(dataset, tokenizer, model_config, batch_size=16, sample_size=None):
    """
    学習用のデータを準備する
    データセットからサンプルを選択し、その場でトークナイズする方式
    
    Args:
        dataset: 元のデータセット
        tokenizer: トークナイザー
        model_config: モデル設定
        batch_size: バッチサイズ
        sample_size: データセットから使用するサンプル数（Noneの場合は全て使用）
        
    Returns:
        訓練用データローダー、評価用データローダー、訓練用データセット、評価用データセット
    """
    from torch.utils.data import DataLoader
    from slm.collator import CustomCollator
    
    print("データを学習用に前処理しています...")
    
    # 訓練データと検証データを取得
    train_data = dataset['train']
    
    # 検証データがあるか確認（様々な名前がある可能性）
    valid_data = None
    for key in ['validation', 'valid', 'test']:
        if key in dataset:
            valid_data = dataset[key]
            print(f"検証データセットを '{key}' から読み込みました")
            break
    
    if valid_data is None:
        # 検証データがない場合は訓練データの10%を使用
        train_size = int(0.9 * len(train_data))
        valid_size = len(train_data) - train_size
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])
    
    # データセットを小さくする（指定された場合）
    if sample_size is not None:
        train_sample_size = min(sample_size, len(train_data))
        valid_sample_size = min(sample_size // 10, len(valid_data))  # 検証データは訓練データの1/10
        
        # トレーニングデータの選択
        if hasattr(train_data, 'select'):
            # datasetsのDatasetオブジェクトの場合
            train_data = train_data.select(range(train_sample_size))
        else:
            # torch.utils.data.Subsetなどの場合
            indices = list(range(train_sample_size))
            train_data = torch.utils.data.Subset(train_data, indices)
            
        # 検証データの選択
        if hasattr(valid_data, 'select'):
            valid_data = valid_data.select(range(valid_sample_size))
        else:
            indices = list(range(valid_sample_size))
            valid_data = torch.utils.data.Subset(valid_data, indices)
        
        print(f"データセットをサンプリング: 学習 {train_sample_size}サンプル, 検証 {valid_sample_size}サンプル")
    
    # カスタムコレーターの初期化（その場でトークナイズするため）
    collator = CustomCollator(
        tokenizer=tokenizer,
        model_config=model_config,
        mlm=True,
        mlm_probability=0.15,
        mask_token_id=tokenizer.mask_token_id,
        qa=False
    )
    
    # データローダーの作成
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator
    )
    
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )
    
    print(f"学習データ: {len(train_data)}サンプル, 検証データ: {len(valid_data)}サンプル")
    
    return train_loader, valid_loader, train_data, valid_data




def get_selected_config():
    """
    本番用設定を返す
    
    Returns:
        設定オブジェクト
    """
    print("本番用設定(config.py)を読み込みます...")
    # configモジュールから直接TrainingConfigを作成
    import slm.config as config_module
    # TrainingConfigを生成
    config = TrainingConfig(
        learning_rate=1e-5,
        batch_size=16, 
        mlm_epochs=3,
        mlm_probability=0.15,
        weight_decay=0.01,
        warmup_steps=500,
        accumulation_steps=1,
        use_amp=True
    )
    
    # 必要な追加属性を設定
    from types import SimpleNamespace
    import datetime
    # 実行時の時刻を取得して一意のrun_nameを生成
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # config.pyの設定値をそのまま使用する
    config_with_dataset = SimpleNamespace(**vars(config))
    # run_nameだけは動的に生成
    config_with_dataset.run_name = f"run_{timestamp}"
    # num_samplesとsample_sizeはこのスクリプト専用の設定
    config_with_dataset.num_samples = 100
    config_with_dataset.sample_size = None
    
    return config_with_dataset


def main():
    """
    Wave Networkモデルの学習と分析処理のメイン関数
    """
    # コマンドライン引数のパース (将来の拡張のために残しておく)
    parser = argparse.ArgumentParser(description='Wave Networkモデルの学習と分析')
    parser.add_argument('--prod', action='store_true', help='本番用設定(config.py)を使用する')
    args = parser.parse_args()
    
    # 設定を読み込む
    selected_config = get_selected_config()
    training_config = selected_config
    
    # 環境設定
    try:
        device, paths_config = setup_environment(selected_config)
        
        # データセットのロード
        dataset = load_dataset_from_disk_or_download(paths_config, selected_config)
        
        # トークナイザーとモデルのセットアップ
        tokenizer, model, model_config = setup_tokenizer_and_model(paths_config, selected_config)
        
        # モデルをデバイスに転送
        model.to(device)
        
        print(f"モデル情報: hidden_size={model_config.hidden_size}, num_layers={model_config.num_layers}, "
              f"vocab_size={model_config.vocab_size}")
    except Exception as e:
        print(f"セットアップ中にエラーが発生しました: {e}")
        raise
    
    # 学習用データの準備
    batch_size = getattr(selected_config, 'batch_size', 16)
    sample_size = getattr(selected_config, 'sample_size', None)
    train_loader, valid_loader, train_dataset, valid_dataset = prepare_data_for_training(
        dataset, tokenizer, model_config, batch_size=batch_size, sample_size=sample_size
    )
    
    # トレーナーの初期化と学習の実行
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        training_config=training_config,  # 設定を使用
        paths_config=paths_config,
        device=device
    )
    
    print("モデル学習を開始します...")
    print(f"MLM学習エポック数: {training_config.mlm_epochs}, Diffusion学習エポック数: {training_config.diffusion_epochs}")
    
    # config.pyの設定に従ってMLM学習を実行（mlm_epochs > 0の場合のみ）
    if training_config.mlm_epochs > 0:
        # MLM学習
        print("\n===== MLM学習フェーズ =====")
        trainer.train_mlm()
        
        # 検証（MLM学習後）
        print("\n===== MLM学習後のバリデーション =====")
        val_loss = trainer.validate()
        print(f"MLM学習後の最終検証結果: Loss={val_loss:.4f}, Perplexity={torch.exp(torch.tensor(val_loss)).item():.2f}")
    else:
        print("\n===== MLM学習をスキップ（mlm_epochs=0） =====")
    
    # config.pyの設定に従ってdiffusion学習を実行（diffusion_epochs > 0の場合のみ）
    if training_config.diffusion_epochs > 0:
        print("\n===== Diffusion学習フェーズ =====")
        trainer.train_diffusion()
        
        # 検証（Diffusion学習後）
        print("\n===== Diffusion学習後のバリデーション =====")
        val_loss = trainer.validate()
        print(f"Diffusion学習後の最終検証結果: Loss={val_loss:.4f}, Perplexity={torch.exp(torch.tensor(val_loss)).item():.2f}")
    else:
        print("\n===== Diffusion学習をスキップ（diffusion_epochs=0） =====") 
    
    print("モデル学習が完了しました！")
    
    # モデルの保存
    try:
        model_path = os.path.join(paths_config.checkpoint_dir, "wave_network_final.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"モデルを {model_path} に保存しました")
    except Exception as e:
        print(f"モデル保存中にエラーが発生しました: {e}")
        print("一時ディレクトリにモデルを保存します...")
        temp_path = os.path.join(os.getcwd(), "model_backup.pt")
        torch.save(model.state_dict(), temp_path)
        print(f"モデルを {temp_path} に保存しました")
    
    # 埋め込みの抽出と可視化 (tools/analysis.pyからインポートした関数を使用)
    try:
        # 可視化用のディレクトリを明示的に作成
        os.makedirs(paths_config.visualization_path, exist_ok=True)
        print(f"可視化用ディレクトリを確認/作成しました: {paths_config.visualization_path}")
        
        num_samples = getattr(selected_config, 'num_samples', 100)
        print(f"埋め込みを抽出して可視化します (サンプル数: {num_samples})...")
        embeddings = extract_embeddings(model, valid_dataset, tokenizer, model_config, device, num_samples=num_samples)
        visualize_embeddings(embeddings, paths_config)
        print("埋め込みの可視化が完了しました")
    except Exception as e:
        print(f"埋め込みの抽出または可視化中にエラーが発生しました: {e}")
        print("学習自体は正常に完了しています。")
    
    print("全ての処理が正常に完了しました！")


if __name__ == '__main__':
    main()