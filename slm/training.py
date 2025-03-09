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
    paths_config = PathsConfig(
        base_dir=getattr(config, "base_dir", os.getcwd()),
        dataset_name=config.dataset_name,
        dataset_subset=config.dataset_subset,
        tokenizer_name=getattr(config, "tokenizer_name", "cl-tohoku/bert-base-japanese-whole-word-masking"),
        output_dir=getattr(config, "output_dir", None),  # Noneを指定して自動生成させる
        cache_dir=getattr(config, "cache_dir", os.path.join(os.getcwd(), "cache")),
        run_name=run_name  # 実行名を渡す（Noneの場合は自動生成）
    )
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
        print(f"データセット {paths_config.dataset_name}/{paths_config.dataset_subset} をロードします...")
        dataset = load_dataset(paths_config.dataset_name, paths_config.dataset_subset, cache_dir=paths_config.cache_dir)
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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=paths_config.cache_dir)
    
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


def merge_configs(easy_config, config_module):
    """
    easy_configの簡易設定とconfig.pyの本番設定をマージする
    easy_configを優先し、存在しない設定はデフォルト値から取得する
    """
    # TrainingConfigで本番用設定を作成
    training_config = TrainingConfig(
        learning_rate=getattr(easy_config, 'learning_rate', 5e-5),
        batch_size=getattr(easy_config, 'batch_size', 16),
        mlm_epochs=getattr(easy_config, 'mlm_epochs', 3),
        mlm_probability=getattr(easy_config, 'mlm_probability', 0.15),
        weight_decay=getattr(easy_config, 'weight_decay', 0.01),
        warmup_steps=getattr(easy_config, 'warmup_steps', 500),
        accumulation_steps=getattr(easy_config, 'accumulation_steps', 1),
        use_amp=getattr(easy_config, 'use_amp', True),
        clip_value=getattr(easy_config, 'clip_value', 1.0)
    )
    
    return training_config


def get_config(config_path=None):
    """
    サンプル設定を生成する関数
    
    Args:
        config_path: 設定ファイルのパス（未実装）
        
    Returns:
        設定オブジェクト（属性アクセス可能なSimpleNamespace）
    """
    from types import SimpleNamespace
    
    # 一意の実行名を生成
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Google Colab環境の検出
    try:
        import IPython
        is_colab = 'google.colab' in str(IPython.get_ipython())
    except (ImportError, NameError):
        is_colab = False
    
    # 実行環境に応じたディレクトリ設定
    if is_colab:
        base_dir = '/content/slm_project'
    else:
        base_dir = os.getcwd()
    
    # デフォルト設定
    config = SimpleNamespace(
        # モデル設定
        hidden_size=768,
        ffn_dim=768 * 4,
        num_layers=3,
        max_seq_len=512,
        dropout_prob=0.1,
        use_rope=True,
        
        # データセット設定
        dataset_name="wikitext",
        dataset_subset="wikitext-2-raw-v1",  # 小さいデータセットで開始
        
        # トークナイザー設定
        model_name="bert-base-uncased",  # 英語用のBERTモデル
        tokenizer_name="bert-base-uncased",  # 英語用のトークナイザー
        
        # 訓練設定
        learning_rate=1e-5,
        batch_size=16,
        mlm_epochs=3,
        diffusion_epochs=2,  # diffusion学習のエポック数も設定
        mlm_probability=0.15,
        weight_decay=0.01,
        warmup_steps=500,
        accumulation_steps=1,
        
        # パス設定
        base_dir=base_dir,
        # output_dirはPathsConfigで自動生成するためNoneに設定
        output_dir=None,
        cache_dir=os.path.join(base_dir, "cache"),
        # 実行名を設定
        run_name=f"run_{timestamp}_{'colab' if is_colab else 'local'}",
        
        # その他の設定
        use_amp=True,
        num_samples=100,
        sample_size=None,  # サンプルサイズ設定（Noneの場合は全データ使用）
    )
    
    return config

def get_selected_config(use_easy_config=True, config_path=None):
    """
    指定された設定タイプに基づいて設定を読み込む
    
    Args:
        use_easy_config: True for easy_config, False for production config
        config_path: Optional path to a config file
    
    Returns:
        設定オブジェクト
    """
    if use_easy_config:
        print("easy_configから簡易設定を読み込みます...")
        if config_path:
            return get_config(config_path)
        else:
            return get_config()
    else:
        print("本番用設定(config.py)を読み込みます...")
        # configモジュールから直接TrainingConfigを作成
        import slm.config as config_module
        # DEFAULT_CONFIGが存在しないので、代わりにデフォルト値を使用してTrainingConfigを生成
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
        return config


def main():
    """
    Wave Networkモデルの学習と分析処理のメイン関数
    コマンドライン引数で設定タイプを選択可能
    """
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='Wave Networkモデルの学習と分析')
    parser.add_argument('--easy', action='store_true', help='easy_configを使用する (デフォルト)')
    parser.add_argument('--prod', action='store_true', help='本番用設定(config.py)を使用する')
    parser.add_argument('--config', type=str, help='設定ファイルへのパス(easy_configの場合のみ)')
    args = parser.parse_args()
    
    # フラグの矛盾をチェック
    if args.easy and args.prod:
        print("警告: --easyと--prodの両方が指定されています。--easyを優先します。")
        use_easy_config = True
    elif args.prod:
        use_easy_config = False
    else:
        # デフォルトはeasy_config
        use_easy_config = True
    
    # 選択された設定を読み込む
    selected_config = get_selected_config(use_easy_config, args.config)
    
    # マージした設定を作成（必要な場合）
    if use_easy_config:
        import slm.config as config_module
        training_config = merge_configs(selected_config, config_module)
    else:
        # 本番用設定の場合は、そのまま使用
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
        training_config=training_config,  # マージした本番用設定を使用
        paths_config=paths_config,
        device=device
    )
    
    print("モデル学習を開始します...")
    
    # MLM学習をスキップし、diffusion学習のみを実行
    if training_config.mlm_epochs > 0:
        # MLM学習（設定されている場合のみ実行）
        print("\n===== MLM学習フェーズ =====")
        trainer.train_mlm()
        
        # 検証（MLM学習後）
        print("\n===== MLM学習後のバリデーション =====")
        val_loss = trainer.validate()
        print(f"MLM学習後の最終検証結果: Loss={val_loss:.4f}, Perplexity={torch.exp(torch.tensor(val_loss)).item():.2f}")
    
    # diffusion学習（training_configにdiffusion_epochsが設定されている場合のみ実行）
    if hasattr(training_config, 'diffusion_epochs') and training_config.diffusion_epochs > 0:
        print("\n===== Diffusion学習フェーズ =====")
        trainer.train_diffusion()
        
        # 検証（Diffusion学習後）
        print("\n===== Diffusion学習後のバリデーション =====")
        val_loss = trainer.validate()
        print(f"Diffusion学習後の最終検証結果: Loss={val_loss:.4f}, Perplexity={torch.exp(torch.tensor(val_loss)).item():.2f}")
    
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