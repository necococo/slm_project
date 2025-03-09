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
    # PathsConfigをconfigの各項目から初期化
    paths_config = PathsConfig(
        base_dir=getattr(config, "base_dir", os.getcwd()),
        dataset_name=config.dataset_name,
        dataset_subset=config.dataset_subset,
        tokenizer_name=getattr(config, "tokenizer_name", "cl-tohoku/bert-base-japanese-whole-word-masking"),
        output_dir=getattr(config, "output_dir", os.path.join(os.getcwd(), "output")),
        cache_dir=getattr(config, "cache_dir", os.path.join(os.getcwd(), "cache"))
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
    print(f"トークナイザー {config.model_name} をロードしています...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=paths_config.cache_dir)
    
    # モデル設定
    model_config = ModelConfig(
        model_name=config.model_name,
        max_seq_len=config.max_seq_len,
        hidden_size=config.hidden_size,
        ffn_dim=config.ffn_dim
    )
    
    # Wave Networkモデルの初期化
    print(f"Wave Networkモデルを初期化しています...")
    model = WaveNetworkLM(model_config)
    
    return tokenizer, model, model_config


def prepare_data_for_training(dataset, tokenizer, model_config, batch_size=16):
    """学習用のデータを準備する"""
    
    print("データを学習用に前処理しています...")
    train_dataset = preprocess_dataset_for_mlm(dataset['train'], tokenizer, model_config.max_seq_len)
    valid_dataset = preprocess_dataset_for_mlm(dataset['validation'], tokenizer, model_config.max_seq_len)
    
    # データローダーの作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    print(f"学習データ: {len(train_dataset)}サンプル, 検証データ: {len(valid_dataset)}サンプル")
    
    return train_loader, valid_loader, train_dataset, valid_dataset


def merge_configs(easy_config, config_module):
    """
    easy_config.pyの簡易設定とconfig.pyの本番設定をマージする
    easy_configを優先し、存在しない設定はconfig_moduleから取得する
    """
    # 本番用設定のデフォルト値
    from slm.config import DEFAULT_CONFIG
    
    # TrainingConfigで本番用設定を作成
    training_config = TrainingConfig(
        learning_rate=getattr(easy_config, 'learning_rate', DEFAULT_CONFIG.get('learning_rate', 5e-5)),
        num_epochs=getattr(easy_config, 'num_epochs', DEFAULT_CONFIG.get('num_epochs', 3)),
        weight_decay=getattr(easy_config, 'weight_decay', DEFAULT_CONFIG.get('weight_decay', 0.01)),
        warmup_steps=getattr(easy_config, 'warmup_steps', DEFAULT_CONFIG.get('warmup_steps', 100)),
        gradient_accumulation_steps=getattr(easy_config, 'gradient_accumulation_steps', 
                                           DEFAULT_CONFIG.get('gradient_accumulation_steps', 1)),
        max_grad_norm=getattr(easy_config, 'max_grad_norm', DEFAULT_CONFIG.get('max_grad_norm', 1.0)),
        logging_steps=getattr(easy_config, 'logging_steps', DEFAULT_CONFIG.get('logging_steps', 10)),
        eval_steps=getattr(easy_config, 'eval_steps', DEFAULT_CONFIG.get('eval_steps', 50)),
        save_steps=getattr(easy_config, 'save_steps', DEFAULT_CONFIG.get('save_steps', 100))
    )
    
    return training_config


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
        # DEFAULT_CONFIGからTrainingConfigを生成
        config = TrainingConfig(**config_module.DEFAULT_CONFIG)
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
    device, paths_config = setup_environment(selected_config)
    
    # データセットのロード
    dataset = load_dataset_from_disk_or_download(paths_config, selected_config)
    
    # トークナイザーとモデルのセットアップ
    tokenizer, model, model_config = setup_tokenizer_and_model(paths_config, selected_config)
    
    # モデルをデバイスに転送
    model.to(device)
    
    # 学習用データの準備
    batch_size = getattr(selected_config, 'batch_size', 16)
    train_loader, valid_loader, train_dataset, valid_dataset = prepare_data_for_training(
        dataset, tokenizer, model_config, batch_size=batch_size
    )
    
    # トレーナーの初期化と学習の実行
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        eval_dataloader=valid_loader,
        device=device,
        config=training_config,  # マージした本番用設定を使用
        paths_config=paths_config
    )
    
    print("モデル学習を開始します...")
    trainer.train()
    print("モデル学習が完了しました！")
    
    # モデルの保存
    save_path = os.path.join(paths_config.model_path, "wave_network_final.pt")
    torch.save(model.state_dict(), save_path)
    print(f"モデルを {save_path} に保存しました")
    
    # 埋め込みの抽出と可視化 (tools/analysis.pyからインポートした関数を使用)
    num_samples = getattr(selected_config, 'num_samples', 100)
    embeddings = extract_embeddings(model, valid_dataset, tokenizer, model_config, device, num_samples=num_samples)
    visualize_embeddings(embeddings, paths_config)
    
    print("全ての処理が正常に完了しました！")


if __name__ == '__main__':
    main()