"""
Wave Network モデルのハイパーパラメータ最適化ツール

Optunaを使用して、Wave Networkモデルの最適なハイパーパラメータを
効率的に探索するためのツールです。MLM (Masked Language Modeling)
タスクのパフォーマンスを最適化します。
"""

import os
import sys
import time
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, Tuple, List, Callable

import optuna
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from slm.config import ModelConfig, TrainingConfig, PathsConfig
from slm.modules.wave_network import WaveNetworkLM
from slm.train import Trainer
from slm.training import (
    setup_environment,
    load_dataset_from_disk_or_download,
    setup_tokenizer_and_model,
    prepare_data_for_training
)

# ロガーの設定
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from types import SimpleNamespace
from typing import Optional, Dict, Any, Union, Tuple, List, Callable, NamedTuple

@dataclass
class HyperparameterSearchConfig:
    """ハイパーパラメータ探索の設定を保持するデータクラス"""
    # 探索するパラメータの範囲
    param_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'complex_init_scale': (0.005, 0.1),    # 複素数初期化スケール
        'dropout_prob': (0.0, 0.5),            # ドロップアウト率
        'learning_rate': (5e-6, 2e-4),         # 学習率
        'warmup_steps': (100, 1000),           # ウォームアップステップ数
        'weight_decay': (0.0, 0.1),            # 重み減衰
        'mlm_probability': (0.1, 0.25),        # MLMマスク確率
    })
    
    # カテゴリカルパラメータ
    categorical_params: Dict[str, List[Any]] = field(default_factory=lambda: {
        'batch_size': [16, 32, 64, 96],        # バッチサイズ候補
        'clip_value': [0.5, 1.0, 2.0],         # 勾配クリップ値の候補
    })
    
    # 探索設定
    n_trials: int = 10                          # 試行回数
    timeout: Optional[int] = None               # タイムアウト（秒）
    n_jobs: int = 1                             # 並列実行数
    sample_size: Optional[int] = None           # サンプリングするデータ数
    sample_ratio: Optional[float] = None        # サンプリング比率
    study_name: Optional[str] = None            # Optunaのスタディ名
    storage: Optional[str] = None               # Optunaのストレージパス
    
    def __post_init__(self):
        """デフォルトのスタディ名とストレージ設定"""
        if self.study_name is None:
            self.study_name = f"wave_network_search_{int(time.time())}"
        
        # ストレージが未指定の場合は後で設定する（パス情報が必要なため）


def subsample_dataset(dataset: Union[Dataset, DatasetDict], 
                     sample_size: Optional[int] = None, 
                     sample_ratio: Optional[float] = None) -> Union[Dataset, DatasetDict]:
    """
    データセットをサブサンプリングする
    
    Args:
        dataset: 元のデータセット
        sample_size: サンプリングするデータ数（指定された場合）
        sample_ratio: サンプリング比率（指定された場合）
        
    Returns:
        サブサンプリングされたデータセット
    """
    if sample_size is None and sample_ratio is None:
        return dataset  # サブサンプリング不要
    
    # DatasetDictの場合
    if isinstance(dataset, dict):
        result = {}
        for split, data in dataset.items():
            if sample_size is not None:
                # 指定サイズでサンプリング
                sample_count = min(sample_size, len(data))
                result[split] = data.select(range(sample_count))
            elif sample_ratio is not None:
                # 指定比率でサンプリング
                sample_count = int(len(data) * sample_ratio)
                result[split] = data.select(range(sample_count))
            else:
                result[split] = data
        return result
    
    # 単一のDatasetの場合
    if sample_size is not None:
        sample_count = min(sample_size, len(dataset))
        return dataset.select(range(sample_count))
    elif sample_ratio is not None:
        sample_count = int(len(dataset) * sample_ratio)
        return dataset.select(range(sample_count))
    
    return dataset


def visualize_optimization_history(study: optuna.study.Study, output_dir: str) -> Dict[str, Any]:
    """
    最適化履歴の可視化と保存を行う
    
    Args:
        study: Optunaのstudyオブジェクト
        output_dir: 可視化結果の保存先ディレクトリ
        
    Returns:
        最適化結果の要約情報
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 最適化履歴をプロット
    plt.figure(figsize=(16, 6))
    
    # パラメータの重要度
    plt.subplot(1, 2, 1)
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('Parameter Importances')
    
    # 最適化の履歴
    plt.subplot(1, 2, 2)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('Optimization History')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optuna_optimization_history.png"))
    
    # パレート図（もしマルチ目的最適化の場合）
    if len(study.objectives) > 1:
        plt.figure(figsize=(10, 8))
        optuna.visualization.matplotlib.plot_pareto_front(study)
        plt.title('Pareto Front')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "optuna_pareto_front.png"))
    
    # パラメータ関係の散布図
    plt.figure(figsize=(15, 15))
    optuna.visualization.matplotlib.plot_slice(study)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optuna_param_slice.png"))
    
    # 結果をJSONとして保存
    best_params = study.best_params
    best_value = study.best_value
    
    results = {
        'best_params': best_params,
        'best_value': float(best_value),  # floatに変換してシリアライズ可能に
        'best_trial': study.best_trial.number,
        'n_trials': len(study.trials)
    }
    
    # JSONファイルに保存
    with open(os.path.join(output_dir, "best_params.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def create_model_from_params(trial: optuna.trial.Trial, 
                            search_config: HyperparameterSearchConfig, 
                            base_config: Dict[str, Any], 
                            device: torch.device) -> Tuple[WaveNetworkLM, AutoTokenizer, ModelConfig, TrainingConfig]:
    """
    トライアルパラメータからモデルと設定を作成する
    
    Args:
        trial: Optunaのトライアルオブジェクト
        search_config: ハイパーパラメータ探索設定
        base_config: 基本設定情報
        device: 計算デバイス
        
    Returns:
        モデル、トークナイザ、モデル設定、トレーニング設定のタプル
    """
    # ハイパーパラメータのサンプリング
    param_ranges = search_config.param_ranges
    categorical_params = search_config.categorical_params
    
    complex_init_scale = trial.suggest_float('complex_init_scale', *param_ranges['complex_init_scale'], log=True)
    dropout_prob = trial.suggest_float('dropout_prob', *param_ranges['dropout_prob'])
    learning_rate = trial.suggest_float('learning_rate', *param_ranges['learning_rate'], log=True)
    warmup_steps = trial.suggest_int('warmup_steps', *param_ranges['warmup_steps'])
    weight_decay = trial.suggest_float('weight_decay', *param_ranges['weight_decay'], log=True)
    mlm_probability = trial.suggest_float('mlm_probability', *param_ranges['mlm_probability'])
    
    # カテゴリカルパラメータ
    batch_size = trial.suggest_categorical('batch_size', categorical_params['batch_size'])
    clip_value = trial.suggest_categorical('clip_value', categorical_params['clip_value'])
    
    # モデル設定の構築
    model_config = ModelConfig(
        hidden_size=base_config.get('hidden_size', 768),
        num_layers=base_config.get('num_layers', 3),
        max_seq_len=base_config.get('max_seq_len', 512),
        dropout_prob=dropout_prob,
        use_rope=base_config.get('use_rope', True),
        complex_init_scale=complex_init_scale,
    )
    
    # トレーニング設定の構築
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        mlm_probability=mlm_probability,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        clip_value=clip_value,
        mlm_epochs=base_config.get('mlm_epochs', 1),
        accumulation_steps=base_config.get('accumulation_steps', 1),
        use_amp=base_config.get('use_amp', True),
    )
    
    # トークナイザの設定
    tokenizer_name = base_config.get('model_name', 'cl-tohoku/bert-base-japanese-whole-word-masking')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model_config.set_tokenizer(tokenizer)
    
    # モデルの構築
    model = WaveNetworkLM(model_config)
    model.to(device)
    
    return model, tokenizer, model_config, training_config


def objective(trial: optuna.trial.Trial, 
             base_config: Dict[str, Any], 
             device: torch.device,
             paths_config: PathsConfig, 
             dataset: Union[Dataset, DatasetDict],
             search_config: HyperparameterSearchConfig) -> float:
    """
    Optunaの目的関数: モデルのトレーニングと評価を行い、評価指標を返す
    
    Args:
        trial: Optunaのトライアル
        base_config: 基本設定
        device: 使用するデバイス
        paths_config: パス設定
        dataset: 使用するデータセット
        search_config: ハイパーパラメータ探索設定
        
    Returns:
        評価損失値 (小さいほど良い)
    """
    try:
        # モデルと設定の作成
        model, tokenizer, model_config, training_config = create_model_from_params(
            trial, search_config, base_config, device
        )
        
        # データの準備
        train_loader, valid_loader, train_dataset, valid_dataset = prepare_data_for_training(
            dataset, tokenizer, model_config, batch_size=training_config.batch_size
        )
        
        # トレーナーの初期化
        trial_dir = os.path.join(paths_config.output_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # トライアル固有のパス設定
        trial_paths = PathsConfig(
            base_dir=trial_dir,
            dataset_name=paths_config.dataset_name,
            dataset_subset=paths_config.dataset_subset,
            tokenizer_name=paths_config.tokenizer_name
        )
        
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_loader,
            eval_dataloader=valid_loader,
            device=device,
            config=training_config,
            paths_config=trial_paths
        )
        
        # モデルのトレーニング（限定されたステップ数）
        logger.info(f"\n===== Trial {trial.number}: トレーニング開始 =====")
        logger.info(f"パラメータ: {trial.params}")
        
        # 最大トレーニングステップ数を制限して早期評価
        max_steps = min(100, len(train_loader))
        trainer.train(max_steps=max_steps)
        
        # 評価
        eval_results = trainer.evaluate()
        perplexity = eval_results.get('perplexity', float('inf'))
        loss = eval_results.get('loss', float('inf'))
        
        # 進捗状況の報告
        logger.info(f"Trial {trial.number} 結果: Perplexity = {perplexity:.4f}, Loss = {loss:.4f}")
        
        # 中間結果を保存
        trial.set_user_attr('perplexity', float(perplexity))
        trial.set_user_attr('loss', float(loss))
        
        # このトライアルの結果を保存
        with open(os.path.join(trial_dir, "metrics.json"), 'w') as f:
            json.dump({
                'perplexity': float(perplexity),
                'loss': float(loss),
                'params': trial.params
            }, f, indent=2)
            
        return loss
        
    except Exception as e:
        logger.error(f"Trial {trial.number} でエラーが発生: {str(e)}", exc_info=True)
        return float('inf')  # エラー時は無限大のロスを返す


def load_base_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    基本設定を読み込む
    
    Args:
        config_file: 設定ファイルのパス
        
    Returns:
        設定情報の辞書
    """
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    
    # デフォルト設定
    return {
        'hidden_size': 768,
        'num_layers': 3,
        'max_seq_len': 512,
        'use_rope': True,
        'mlm_epochs': 1,
        'accumulation_steps': 1,
        'use_amp': True,
        'model_name': 'cl-tohoku/bert-base-japanese-whole-word-masking',
    }


def load_and_prepare_dataset(paths_config: PathsConfig, 
                            base_config: Dict[str, Any],
                            sample_size: Optional[int] = None,
                            sample_ratio: Optional[float] = None) -> Union[Dataset, DatasetDict]:
    """
    データセットを読み込み、準備する
    
    Args:
        paths_config: パス設定
        base_config: 基本設定
        sample_size: サンプリングサイズ
        sample_ratio: サンプリング比率
        
    Returns:
        処理済みデータセット
    """
    # 前処理済みデータの確認
    preprocessed_path = os.path.join(paths_config.data_dir, "processed_raw")
    if os.path.exists(preprocessed_path):
        logger.info("トークナイズ済みデータセットを読み込みます...")
        full_dataset = load_from_disk(preprocessed_path)
    else:
        logger.info("生データセットを読み込み、処理します...")
        full_dataset = load_dataset_from_disk_or_download(paths_config, base_config)
    
    # データセットのサブサンプリング
    return subsample_dataset(full_dataset, sample_size, sample_ratio)


def dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    """
    辞書をSimpleNamespace（属性アクセス可能なオブジェクト）に変換する
    
    Args:
        d: 変換する辞書
    
    Returns:
        属性アクセス可能なオブジェクト
    """
    # 再帰的に処理するため、ネストされた辞書も変換
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return SimpleNamespace(**d)


def ensure_required_config_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    setup_environment関数に必要な設定キーが存在することを確認し、
    存在しない場合はデフォルト値を設定する
    
    Args:
        config: 設定辞書
    
    Returns:
        必要なキーを持つ設定辞書
    """
    # 必須とデフォルト値のマッピング
    required_keys = {
        'dataset_name': 'singletongue/wikipedia-utils',
        'dataset_subset': 'corpus-jawiki-20230403-filtered-large',
        'model_name': 'cl-tohoku/bert-base-japanese-whole-word-masking',
        'tokenizer_name': config.get('model_name', 'cl-tohoku/bert-base-japanese-whole-word-masking'),
        'base_dir': os.path.join(os.getcwd(), 'slm_output'),
        'output_dir': os.path.join(os.getcwd(), 'slm_output'),
        'cache_dir': os.path.join(os.getcwd(), 'cache'),
    }
    
    # 存在しないキーを追加
    for key, default_value in required_keys.items():
        if key not in config:
            config[key] = default_value
    
    return config


def run_hyperparameter_search(
    config_file: Optional[str] = None, 
    n_trials: int = 10, 
    timeout: Optional[int] = None, 
    n_jobs: int = 1,
    sample_size: Optional[int] = None,
    sample_ratio: Optional[float] = None
) -> Tuple[optuna.study.Study, Dict[str, Any]]:
    """
    ハイパーパラメータ探索のメイン関数
    
    Args:
        config_file: 設定ファイルのパス
        n_trials: 試行回数
        timeout: タイムアウト（秒）
        n_jobs: 並列実行数
        sample_size: サンプリングするデータ数
        sample_ratio: サンプリング比率
        
    Returns:
        (study, results): Optunaのstudyオブジェクトと結果の辞書
    """
    logger.info("Wave Network モデルのハイパーパラメータ探索を開始します")
    
    # 基本設定の読み込み
    base_config = load_base_config(config_file)
    
    # 必須キーを持つことを確認
    base_config = ensure_required_config_keys(base_config)
    
    # 辞書を属性アクセス可能なオブジェクトに変換
    config_obj = dict_to_namespace(base_config)
    
    # 環境設定 - 変換したオブジェクトを渡す
    device, paths_config = setup_environment(config_obj)
    
    # 探索設定の作成
    search_config = HyperparameterSearchConfig(
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        sample_size=sample_size,
        sample_ratio=sample_ratio
    )
    
    # デバイス設定は setup_environment で取得済み
    logger.info(f"Using device: {device}")
    
    # パスの設定は既に paths_config に含まれている
    # ストレージパスの設定
    os.makedirs(paths_config.base_dir, exist_ok=True)
    search_config.storage = f"sqlite:///{paths_config.base_dir}/optuna_{search_config.study_name}.db"
    
    # データセット準備
    dataset = load_and_prepare_dataset(
        paths_config, base_config, sample_size=sample_size, sample_ratio=sample_ratio
    )
    
    # Optuna Studyの作成
    study = optuna.create_study(
        study_name=search_config.study_name,
        storage=search_config.storage,
        load_if_exists=True,
        direction="minimize",  # 損失を最小化
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    # 探索の実行
    objective_func = lambda trial: objective(
        trial, base_config, device, paths_config, dataset, search_config
    )
    
    study.optimize(
        objective_func,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs
    )
    
    # 結果の可視化と保存
    results = visualize_optimization_history(study, paths_config.base_dir)
    
    logger.info("\n===== ハイパーパラメータ探索結果 =====")
    logger.info(f"最適パラメータ: {results['best_params']}")
    logger.info(f"最小損失値: {results['best_value']:.4f}")
    logger.info(f"総試行回数: {results['n_trials']}")
    logger.info(f"結果保存先: {paths_config.base_dir}")
    
    return study, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wave Network モデルのハイパーパラメータ探索")
    parser.add_argument("--config", type=str, help="基本設定を含むJSONファイルのパス")
    parser.add_argument("--n-trials", type=int, default=10, help="Optuna探索の試行回数")
    parser.add_argument("--timeout", type=int, default=None, help="タイムアウト（秒）")
    parser.add_argument("--n-jobs", type=int, default=1, help="並列実行数")
    parser.add_argument("--sample-size", type=int, default=None, help="データサンプリングサイズ")
    parser.add_argument("--sample-ratio", type=float, default=None, help="データサンプリング比率")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", 
                        help="ログレベル設定")
    
    args = parser.parse_args()
    
    # ログレベル設定
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    run_hyperparameter_search(
        config_file=args.config,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        sample_size=args.sample_size,
        sample_ratio=args.sample_ratio
    )
