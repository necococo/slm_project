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
import math
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from slm.config import ModelConfig, TrainingConfig, PathsConfig
from slm.modules.wave_network import WaveNetworkLM
# トレーナーのインポート
import importlib.util
import sys

# 明示的にトレーナーモジュールをロード
trainer_module_path = os.path.join(project_root, 'slm', 'train.py')
spec = importlib.util.spec_from_file_location("train_module", trainer_module_path)
train_module = importlib.util.module_from_spec(spec)
sys.modules["train_module"] = train_module
spec.loader.exec_module(train_module)
Trainer = train_module.Trainer

from slm.training import (
    setup_environment,
    load_dataset_from_disk_or_download,
    setup_tokenizer_and_model,
)

# prepare_data_for_trainingを自前で実装（training.pyのものは使わない）
def prepare_data_for_training(dataset, tokenizer, model_config, batch_size=16, sample_size=None, training_config=None):
    """
    学習用のデータを準備する
    
    Args:
        dataset: 元のデータセット
        tokenizer: トークナイザー
        model_config: モデル設定
        batch_size: バッチサイズ
        sample_size: データセットから使用するサンプル数（Noneの場合は全て使用）
        training_config: トレーニング設定（MLM確率などを含む）
        
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
    
    # データセットを前処理（トークン化）
    def preprocess_dataset(dataset):
        from torch.utils.data import Dataset as TorchDataset
        
        # torch.utils.data.Subsetの場合、.map()メソッドがないので処理を変える
        if isinstance(dataset, TorchDataset) and not hasattr(dataset, 'map'):
            # torch.utils.data.Subsetの場合
            original_dataset = dataset.dataset
            indices = dataset.indices
            processed_examples = []
            
            max_seq_len = model_config.max_seq_len
            
            for idx in indices:
                item = original_dataset[idx]
                
                # text属性があるか確認
                if hasattr(item, 'text') or (isinstance(item, dict) and 'text' in item):
                    text = item.text if hasattr(item, 'text') else item['text']
                    
                    # トークン化
                    encoded = tokenizer(
                        text, 
                        truncation=True,
                        max_length=max_seq_len,
                        padding='max_length',
                        return_tensors='pt'
                    )
                    
                    # スクイーズしてバッチ次元を削除
                    for k, v in encoded.items():
                        encoded[k] = v.squeeze(0)
                        
                    # テンソルをリストに変換（Collatorはリストを期待）
                    encoded['input_ids'] = encoded['input_ids'].tolist()
                    encoded['attention_mask'] = encoded['attention_mask'].tolist()
                    
                    processed_examples.append(encoded)
                else:
                    # 既に前処理されている場合はそのまま追加
                    processed_examples.append(item)
            
            # 新しいデータセットを作成
            from torch.utils.data import TensorDataset
            
            class SimpleDataset(TorchDataset):
                def __init__(self, examples):
                    self.examples = examples
                
                def __len__(self):
                    return len(self.examples)
                
                def __getitem__(self, idx):
                    return self.examples[idx]
            
            return SimpleDataset(processed_examples)
        else:
            # datasetsのDatasetオブジェクトの場合
            # 'text'フィールドがあるか確認
            if 'text' in dataset.features:
                def tokenize_function(examples):
                    return tokenizer(
                        examples['text'],
                        truncation=True,
                        max_length=model_config.max_seq_len,
                        padding='max_length'
                    )
                return dataset.map(tokenize_function, batched=True)
            # 既に前処理されている場合は何もしない
            return dataset
    
    # データセットの前処理
    processed_train_data = preprocess_dataset(train_data)
    processed_valid_data = preprocess_dataset(valid_data)
    
    # カスタムコレーターの初期化
    mlm_probability = 0.15  # デフォルト値
    if training_config is not None and hasattr(training_config, 'mlm_probability'):
        mlm_probability = training_config.mlm_probability
    
    collator = CustomCollator(
        tokenizer=tokenizer,
        model_config=model_config,
        mlm=True,
        mlm_probability=mlm_probability,  # トライアルパラメータまたはデフォルト値
        mask_token_id=tokenizer.mask_token_id,
        qa=False
    )
    
    # データローダーの作成
    train_loader = DataLoader(
        processed_train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator
    )
    
    valid_loader = DataLoader(
        processed_valid_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )
    
    print(f"学習データ: {len(processed_train_data)}サンプル, 検証データ: {len(processed_valid_data)}サンプル")
    
    return train_loader, valid_loader, processed_train_data, processed_valid_data

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
        'learning_rate': (2e-5, 1e-4),         # 学習率の範囲を制限
        'weight_decay': (1e-4, 1e-4),          # 固定値 0.0001
    })
    
    # 整数パラメータの範囲
    int_ranges: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        'warmup_steps': (500, 500),            # 固定値 500
    })
    
    # カテゴリカルパラメータ
    categorical_params: Dict[str, List[Any]] = field(default_factory=lambda: {
        'num_layers': [1, 3],                  # Wave Networkのレイヤー数を1または3に制限
        'batch_size': [4, 8, 16],              # バッチサイズ候補を制限
        'clip_value': [1.0],                   # 勾配クリップ値を固定
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
                if split != 'train':
                    # 検証/テストセットは小さくする
                    sample_count = min(sample_count // 10, len(data))
                
                logger.info(f"'{split}'セットを{len(data)}から{sample_count}サンプルにサブサンプリングします")
                result[split] = data.select(range(sample_count))
            elif sample_ratio is not None:
                # 指定比率でサンプリング
                sample_count = int(len(data) * sample_ratio)
                if split != 'train':
                    # 検証/テストセットは小さくする（訓練データの1/10）
                    sample_count = int(sample_count * 0.1)
                
                logger.info(f"'{split}'セットを{len(data)}から{sample_count}サンプルにサブサンプリングします")
                result[split] = data.select(range(sample_count))
            else:
                result[split] = data
        return result
    
    # 単一のDatasetの場合
    if sample_size is not None:
        sample_count = min(sample_size, len(dataset))
        logger.info(f"データセットを{len(dataset)}から{sample_count}サンプルにサブサンプリングします")
        return dataset.select(range(sample_count))
    elif sample_ratio is not None:
        sample_count = int(len(dataset) * sample_ratio)
        logger.info(f"データセットを{len(dataset)}から{sample_count}サンプルにサブサンプリングします")
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
    # study.objectivesではなくstudy.directionsの長さで判断
    if hasattr(study, 'directions') and len(study.directions) > 1:
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
    int_ranges = search_config.int_ranges
    categorical_params = search_config.categorical_params
    
    # 連続パラメータのサンプリング（固定値以外）
    learning_rate = trial.suggest_float('learning_rate', *param_ranges['learning_rate'], log=True)
    warmup_steps = trial.suggest_int('warmup_steps', *param_ranges['warmup_steps'])
    weight_decay = trial.suggest_float('weight_decay', *param_ranges['weight_decay'], log=True)
    mlm_probability = trial.suggest_float('mlm_probability', *param_ranges['mlm_probability'])  # 対数スケールなし
    
    # 固定値の設定
    complex_init_scale = 0.02  # 固定値
    dropout_prob = 0.2  # 固定値
    noise_std = 0.1  # 固定値: 生体ゆらぎの標準偏差を0.1に固定
    
    # カテゴリカルパラメータ
    num_layers = trial.suggest_categorical('num_layers', categorical_params['num_layers'])  # [1, 3, 6]から選択
    batch_size = trial.suggest_categorical('batch_size', categorical_params['batch_size'])
    clip_value = trial.suggest_categorical('clip_value', categorical_params['clip_value'])
    
    # 固定のカテゴリカルパラメータ
    use_bio_noise = True  # 固定値: 生体ゆらぎを使用
    trainable_noise = False  # 固定値: ノイズスケールは固定（学習しない）
    use_wavelet = True  # 固定値: ウェーブレット変換を使用
    
    # モデル設定の構築 - 生体ゆらぎとウェーブレットのパラメータを含める
    model_config = ModelConfig(
        hidden_size=base_config.get('hidden_size', 768),
        num_layers=num_layers,  # サンプリングされたレイヤー数を使用 [1, 3, 6]
        max_seq_len=base_config.get('max_seq_len', 512),
        dropout_prob=dropout_prob,
        use_rope=base_config.get('use_rope', True),
        complex_init_scale=complex_init_scale,
        # 生体ゆらぎ関連のパラメータ（固定値）
        use_bio_noise=use_bio_noise,  # True
        noise_std=noise_std,          # 0.01
        trainable_noise=trainable_noise,  # False
        # ウェーブレット関連のパラメータ（固定値）
        use_wavelet=use_wavelet,      # True
        wavelet_name="haar",
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
    
    # tqdmを使用するための準備
    try:
        from tqdm.auto import tqdm
    except ImportError:
        # tqdmがない場合はダミーのtqdmを作成
        def tqdm(iterable, **kwargs):
            return iterable
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
        
        # データの準備（サンプルサイズとMLM確率を指定）
        train_loader, valid_loader, train_dataset, valid_dataset = prepare_data_for_training(
            dataset, 
            tokenizer, 
            model_config, 
            batch_size=training_config.batch_size,
            sample_size=search_config.sample_size,
            training_config=training_config  # MLM確率を含むトレーニング設定を渡す
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
        
        # MLM確率をトライアルパラメータから設定
        if 'mlm_probability' in trial.params:
            training_config.mlm_probability = trial.params['mlm_probability']
            
        # トークナイザーをモデルの設定に追加（Trainerのtrain_mlmメソッドで必要）
        model.config.tokenizer = tokenizer
        
        # Trainerクラスの初期化パラメータに合わせて修正
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            training_config=training_config,
            paths_config=trial_paths,
            device=device
        )
        
        # モデルのトレーニング（限定されたステップ数）
        logger.info(f"\n===== Trial {trial.number}: トレーニング開始 =====")
        logger.info(f"パラメータ: {trial.params}")
        logger.info(f"データセットサイズ: 訓練={len(train_dataset)}, 検証={len(valid_dataset)}")
        
        # トレーニングデータを小さくしてメモリ使用量を削減
        # オプションで自動混合精度を有効化
        if not hasattr(training_config, 'use_amp'):
            training_config.use_amp = True
            
        # 最大トレーニングステップ数を制限して早期評価
        max_steps = min(50, len(train_loader))
        
        # メモリを解放してからトレーニング開始
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # MLM学習をスキップしてdiffusion学習のみを実行
        print("\n===== Diffusion学習フェーズ =====")
        # MLM学習は実行しない
        # trainer.train_mlm(num_epochs=1)
        
        # Diffusion学習を実行
        training_config.diffusion_epochs = 2  # 探索では短いエポック数で実行
        trainer.train_diffusion(num_epochs=2)
        
        # 評価
        # Trainerクラスのvalidateメソッドを呼び出す
        try:
            print("\n===== バリデーションフェーズ =====")
            loss = trainer.validate()
            # 損失値からperplexityを計算（損失値が非常に大きい場合は無限大とする）
            perplexity = float('inf') if loss > 20 else torch.exp(torch.tensor(loss)).item()
        except Exception as e:
            logger.error(f"評価中にエラーが発生: {str(e)}")
            loss = float('inf')
            perplexity = float('inf')
        
        # 進捗状況の報告
        logger.info(f"Trial {trial.number} 結果: [Validation] Perplexity = {perplexity:.4f}, Loss = {loss:.4f}")
        
        # 中間結果を保存
        trial.set_user_attr('perplexity', float(perplexity))
        trial.set_user_attr('loss', float(loss))
        trial.set_user_attr('dataset_size', len(train_dataset))
        
        # このトライアルの結果を保存
        with open(os.path.join(trial_dir, "metrics.json"), 'w') as f:
            json.dump({
                'perplexity': float(perplexity),
                'loss': float(loss),
                'params': trial.params,
                'dataset_size': len(train_dataset)
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
    if (config_file and os.path.exists(config_file)):
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
    
    # データセットの構造を出力（デバッグ用）
    logger.info(f"データセット構造: {list(full_dataset.keys())}")
    
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
    # Google Driveのマウント先パスを確認
    drive_base_path = '/content/drive/MyDrive/slm'
    
    # 現在時刻をもとにした結果保存用ディレクトリ名を作成
    timestamp = int(time.time())
    results_dir_name = f"layer_search_biowavelet_{timestamp}"  # レイヤー数探索実験用にスタディ名を変更
    
    # Google Driveがマウントされているか確認
    if os.path.exists('/content/drive/MyDrive'):
        # Google Driveのslmディレクトリを作成（存在しない場合）
        os.makedirs(drive_base_path, exist_ok=True)
        
        # 結果保存用ディレクトリパスを作成
        results_path = os.path.join(drive_base_path, results_dir_name)
        print(f"ハイパーパラメータ探索の結果を {results_path} に保存します")
    else:
        # Google Driveがマウントされていない場合はローカルディレクトリを使用
        results_path = os.path.join(os.getcwd(), f'slm_output_{results_dir_name}')
        print(f"Google Driveが見つかりません。結果を {results_path} に保存します")
    
    # 必須とデフォルト値のマッピング
    required_keys = {
        'dataset_name': 'wikitext',
        'dataset_subset': 'wikitext-2-raw-v1',  # 小さいデータセットで開始
        'model_name': 'bert-base-uncased',  # 英語用のBERTモデル
        'tokenizer_name': config.get('model_name', 'bert-base-uncased'),  # 英語用のトークナイザー
        'base_dir': results_path,  # Google Drive上のパスに変更
        'output_dir': results_path,  # Google Drive上のパスに変更
        'cache_dir': os.path.join(os.getcwd(), 'cache'),  # キャッシュはローカルに残す
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
    logger.info(f"データセット設定: sample_size={sample_size}, sample_ratio={sample_ratio}")
    
    # パスの設定は既に paths_config に含まれている
    # ストレージパスの設定
    os.makedirs(paths_config.base_dir, exist_ok=True)
    search_config.storage = f"sqlite:///{paths_config.base_dir}/optuna_{search_config.study_name}.db"
    
    # データセット準備（選択対象のデータセットをロード）
    # データセットを小さめにロード（サンプルサイズを直接使用）
    # 明示的にサンプルサイズを渡してデータセットをロード
    sample_size_to_use = search_config.sample_size or 1000  # デフォルト1000
    sample_ratio_to_use = search_config.sample_ratio
    
    print(f"データセットをロード中: サンプルサイズ={sample_size_to_use}, サンプル比率={sample_ratio_to_use}")
    
    dataset = load_and_prepare_dataset(
        paths_config, base_config, sample_size=sample_size_to_use, sample_ratio=sample_ratio_to_use
    )
    
    # 検証データセットが存在するかチェック（validation, valid, testなど様々な名前がある可能性）
    validation_key = None
    for key in ['validation', 'valid', 'test']:
        if key in dataset:
            validation_key = key
            break
    
    logger.info(f"ロードされたデータセット: train={len(dataset['train'])}, "
                f"validation={validation_key and len(dataset[validation_key]) or 'なし'}")
    
    # 実際のサンプリングはprepare_data_for_trainingの中で実行される
    
    # Optuna Studyの作成
    study = optuna.create_study(
        study_name=search_config.study_name,
        storage=search_config.storage,
        load_if_exists=True,
        direction="minimize",  # 損失を最小化
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    # CUDA メモリ管理設定
    if torch.cuda.is_available():
        # メモリフラグメンテーション対策として expandable_segments を有効化
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("CUDA memory management configuration set: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        
        # メモリを一度解放
        torch.cuda.empty_cache()
    
    # 探索の実行
    objective_func = lambda trial: objective(
        trial, base_config, device, paths_config, dataset, search_config
    )
    
    # tqdmのインポートを確保
    try:
        from tqdm.auto import tqdm
        import optuna.integration
        optuna_tqdm = optuna.integration.TQDMProgressBar()
        callbacks = [optuna_tqdm]
    except (ImportError, AttributeError):
        # tqdmまたはoptuna.integrationがない場合
        callbacks = []
    
    study.optimize(
        objective_func,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        callbacks=callbacks
    )
    
    # 結果の可視化と保存
    results = visualize_optimization_history(study, paths_config.base_dir)
    
    logger.info("\n===== ハイパーパラメータ探索結果 =====")
    logger.info(f"最適パラメータ: {results['best_params']}")
    logger.info(f"最小損失値: {results['best_value']:.4f}")
    logger.info(f"総試行回数: {results['n_trials']}")
    logger.info(f"結果保存先: {paths_config.base_dir}")
    
    return study, results


def check_and_mount_google_drive():
    """
    Google Colab環境でGoogle Driveがマウントされているか確認し、
    マウントされていない場合はマウントを促すメッセージを表示します。
    """
    if not os.path.exists('/content/drive/MyDrive'):
        print("\n" + "=" * 80)
        print("Google Driveがマウントされていません。")
        print("Google Colab環境でこのスクリプトを実行している場合は、")
        print("以下のコードを実行してGoogle Driveをマウントしてください：")
        print("\nfrom google.colab import drive")
        print("drive.mount('/content/drive')")
        print("=" * 80 + "\n")
        return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wave Network モデルのハイパーパラメータ探索")
    parser.add_argument("--config", type=str, help="基本設定を含むJSONファイルのパス")
    parser.add_argument("--n-trials", type=int, default=10, help="Optuna探索の試行回数")
    parser.add_argument("--timeout", type=int, default=None, help="タイムアウト（秒）")
    parser.add_argument("--n-jobs", type=int, default=1, help="並列実行数")
    parser.add_argument("--sample-size", type=int, default=1000, help="データサンプリングサイズ (デフォルト: 1000)")
    parser.add_argument("--sample-ratio", type=float, default=None, help="データサンプリング比率")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", 
                        help="ログレベル設定")
    parser.add_argument("--force-local", action="store_true", help="Google Driveが利用可能でも結果をローカルに保存")
    
    args = parser.parse_args()
    
    # 引数の確認
    if args.sample_size is not None and args.sample_ratio is not None:
        print("警告: --sample-sizeと--sample-ratioの両方が指定されています。--sample-sizeを優先します。")
        args.sample_ratio = None
    
    # ログレベル設定
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # サンプルサイズの情報をログに出力
    if args.sample_size:
        logger.info(f"データセットサンプルサイズ: {args.sample_size}サンプル")
    elif args.sample_ratio:
        logger.info(f"データセットサンプル比率: {args.sample_ratio * 100:.1f}%")
    else:
        logger.info("サンプリングなし: フルデータセットを使用")
    
    # Google Driveのマウント確認（強制ローカル保存オプションがない場合）
    if not args.force_local:
        check_and_mount_google_drive()
    
    # ハイパーパラメータ探索に関する情報を出力
    logger.info("生体ゆらぎ+ウェーブレット変換の層数探索を実行します")
    logger.info("固定パラメータ:")
    logger.info("- use_bio_noise: True (生体ゆらぎ機能を使用)")
    logger.info("- noise_std: 0.01 (生体ゆらぎの強度)")
    logger.info("- trainable_noise: False (ノイズスケールは固定)")
    logger.info("- use_wavelet: True (ウェーブレット変換を使用)")
    logger.info("- wavelet_name: haar (Haarウェーブレットを使用)")
    logger.info("探索パラメータ:")
    logger.info("- num_layers: [1, 3, 6] (Wave Networkの層数)")
    
    run_hyperparameter_search(
        config_file=args.config,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        sample_size=args.sample_size,
        sample_ratio=args.sample_ratio
    )
