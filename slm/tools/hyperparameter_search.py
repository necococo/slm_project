"""
Wave Network モデルのハイパーパラメータ最適化ツール
Optunaを使用して最適なパラメータを探索します。
"""

import os
import sys
import time
import json
import argparse
import optuna
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer

# パスを追加してプロジェクト内のモジュールをインポートできるようにする
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from slm.config import ModelConfig, TrainingConfig, PathsConfig
from slm.modules.wave_network import WaveNetworkLM
from slm.train import Trainer
from slm.training import (
    setup_environment,
    load_dataset_from_disk_or_download,
    setup_tokenizer_and_model,
    prepare_data_for_training
)

# 探索するパラメータの定義
PARAM_RANGES = {
    # モデル構造関連パラメータ
    'complex_init_scale': (0.005, 0.1),    # 複素数初期化スケール
    'dropout_prob': (0.0, 0.5),            # ドロップアウト率
    
    # 学習設定関連パラメータ
    'learning_rate': (5e-6, 2e-4),         # 学習率
    'warmup_steps': (100, 1000),           # ウォームアップステップ数
    'weight_decay': (0.0, 0.1),            # 重み減衰
    'mlm_probability': (0.1, 0.25),        # MLMマスク確率
}

# オプションのパラメータと固定値
OPTIONAL_PARAMS = {
    'batch_size': [16, 32, 64, 96],        # バッチサイズ候補
    'clip_value': [0.5, 1.0, 2.0],         # 勾配クリップ値の候補
}

def visualize_optimization_history(study, output_dir):
    """最適化履歴の可視化と保存を行う"""
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

def objective(trial, base_config, device, paths_config, dataset):
    """Optunaの目的関数: モデルのトレーニングと評価を行い、評価指標を返す"""
    # ハイパーパラメータのサンプリング
    complex_init_scale = trial.suggest_float('complex_init_scale', *PARAM_RANGES['complex_init_scale'], log=True)
    dropout_prob = trial.suggest_float('dropout_prob', *PARAM_RANGES['dropout_prob'])
    learning_rate = trial.suggest_float('learning_rate', *PARAM_RANGES['learning_rate'], log=True)
    warmup_steps = trial.suggest_int('warmup_steps', *PARAM_RANGES['warmup_steps'])
    weight_decay = trial.suggest_float('weight_decay', *PARAM_RANGES['weight_decay'], log=True)
    mlm_probability = trial.suggest_float('mlm_probability', *PARAM_RANGES['mlm_probability'])
    
    # オプションのパラメータ
    batch_size = trial.suggest_categorical('batch_size', OPTIONAL_PARAMS['batch_size'])
    clip_value = trial.suggest_categorical('clip_value', OPTIONAL_PARAMS['clip_value'])
    
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
        # その他のパラメータは基本設定から取得
        mlm_epochs=base_config.get('mlm_epochs', 1),
        accumulation_steps=base_config.get('accumulation_steps', 1),
        use_amp=base_config.get('use_amp', True),
    )
    
    # モデルのセットアップ
    tokenizer = AutoTokenizer.from_pretrained(base_config.get('model_name', 'cl-tohoku/bert-base-japanese-whole-word-masking'))
    model_config.set_tokenizer(tokenizer)  # tokenizer情報をセット
    
    model = WaveNetworkLM(model_config)
    model.to(device)
    
    # データの準備
    train_loader, valid_loader, train_dataset, valid_dataset = prepare_data_for_training(
        dataset, tokenizer, model_config, batch_size=batch_size
    )
    
    # トレーナーの初期化
    trial_dir = os.path.join(paths_config.output_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # 一時的なPathsConfigを作成（各トライアルごとに出力先を分ける）
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
    
    # モデルのトレーニング（1エポックのみ）と評価
    print(f"\n===== Trial {trial.number}: トレーニング開始 =====")
    print(f"パラメータ: {trial.params}")
    
    # トレーニングと評価
    try:
        # 最大トレーニングステップ数を制限して早期評価
        max_steps = min(100, len(train_loader))  # 最初の100ステップまたはデータセットサイズ
        trainer.train(max_steps=max_steps)
        
        # 評価
        eval_results = trainer.evaluate()
        perplexity = eval_results.get('perplexity', float('inf'))
        loss = eval_results.get('loss', float('inf'))
        
        # 進捗状況の報告
        print(f"Trial {trial.number} 結果: Perplexity = {perplexity:.4f}, Loss = {loss:.4f}")
        
        # 中間結果を保存
        trial.set_user_attr('perplexity', float(perplexity))
        trial.set_user_attr('loss', float(loss))
        
        # このトライアルの結果を可視化
        with open(os.path.join(trial_dir, "metrics.json"), 'w') as f:
            json.dump({
                'perplexity': float(perplexity),
                'loss': float(loss),
                'params': trial.params
            }, f, indent=2)
            
        # 最適化対象の指標を返す (損失値または特定の評価指標)
        return loss
        
    except Exception as e:
        print(f"Trial {trial.number} でエラーが発生: {e}")
        return float('inf')  # エラー時は無限大のロスを返す

def run_hyperparameter_search(config_file=None, n_trials=10, timeout=None, n_jobs=1):
    """ハイパーパラメータ探索のメイン関数"""
    print("Wave Network モデルのハイパーパラメータ探索を開始します")
    
    # 基本設定を読み込む
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            base_config = json.load(f)
    else:
        # デフォルト設定
        base_config = {
            'hidden_size': 768,
            'num_layers': 3,
            'max_seq_len': 512,
            'use_rope': True,
            'mlm_epochs': 1,
            'accumulation_steps': 1,
            'use_amp': True,
            'model_name': 'cl-tohoku/bert-base-japanese-whole-word-masking',
        }
    
    # 環境設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 設定クラスのインスタンス化
    paths_config = PathsConfig(
        base_dir=base_config.get('output_dir', '/tmp/wave_network_optuna'),
        dataset_name=base_config.get('dataset_name', 'singletongue/wikipedia-utils'),
        dataset_subset=base_config.get('dataset_subset', 'corpus-jawiki-20230403-filtered-large')
    )
    
    # データセットをロード
    print("データセットをロードしています...")
    dataset = load_dataset_from_disk_or_download(paths_config, base_config)
    
    # Optuna Studyの作成
    study_name = f"wave_network_search_{int(time.time())}"
    storage_name = f"sqlite:///{paths_config.base_dir}/optuna_{study_name}.db"
    os.makedirs(paths_config.base_dir, exist_ok=True)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",  # 損失を最小化
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    # 探索の実行
    study.optimize(
        lambda trial: objective(trial, base_config, device, paths_config, dataset),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs
    )
    
    # 結果の可視化と保存
    results = visualize_optimization_history(study, paths_config.base_dir)
    
    print("\n===== ハイパーパラメータ探索結果 =====")
    print(f"最適パラメータ: {results['best_params']}")
    print(f"最小損失値: {results['best_value']:.4f}")
    print(f"総試行回数: {results['n_trials']}")
    print(f"結果保存先: {paths_config.base_dir}")
    
    return study, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wave Network モデルのハイパーパラメータ探索")
    parser.add_argument("--config", type=str, help="基本設定を含むJSONファイルのパス")
    parser.add_argument("--n-trials", type=int, default=10, help="Optuna探索の試行回数")
    parser.add_argument("--timeout", type=int, default=None, help="タイムアウト（秒）")
    parser.add_argument("--n-jobs", type=int, default=1, help="並列実行数")
    args = parser.parse_args()
    
    run_hyperparameter_search(
        config_file=args.config,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs
    )
