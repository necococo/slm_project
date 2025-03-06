# utils.py
import torch
import os
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, Union

def save_checkpoint(state: dict, checkpoint_dir: str, filename: str = None):
    """
    チェックポイントを保存する関数。
    
    Args:
        state (dict): モデル、オプティマイザ、その他情報を含む辞書。
        checkpoint_dir (str): チェックポイントを保存するディレクトリ。
        filename (str, optional): 保存するファイル名。指定がない場合は自動的に "checkpoint_epoch_{epoch}.pt" とします。
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if filename is None:
        epoch = state.get("epoch", 0)
        filename = f"checkpoint_epoch_{epoch}.pt"
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(checkpoint_path: str, model: nn.Module) -> None:
    """
    保存されたチェックポイントからモデルを読み込みます。
    
    Args:
        checkpoint_path: チェックポイントファイルのパス
        model: 読み込み先のモデル
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"チェックポイントを読み込みました: {checkpoint_path}")
    except Exception as e:
        print(f"チェックポイント読み込み中にエラーが発生しました: {e}")

def get_model_size(model: nn.Module) -> int:
    """
    モデルのパラメータ数を計算します。
    
    Args:
        model: 計算対象のモデル
        
    Returns:
        パラメータの総数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_flops_per_batch(model: nn.Module, input_shape: Tuple[int, int]) -> float:
    """
    1バッチあたりのおおよそのFLOPsを計算（簡易的な推定）
    
    Args:
        model: 対象モデル
        input_shape: 入力形状 (batch_size, seq_len)
        
    Returns:
        推定されるFLOPs数（浮動小数点演算数）
    """
    # 非常に簡易的な推定。実際には各層ごとに正確な計算をするべき
    batch_size, seq_len = input_shape
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_layers
    vocab_size = model.config.vocab_size
    
    # 埋め込み層のFLOPs
    embedding_flops = batch_size * seq_len * hidden_size
    
    # 各トランスフォーマー層のFLOPs (非常に簡易的)
    # Wave Networkはself-attentionを使わないので、単純に各層のFFNのFLOPsだけ考慮
    layer_flops = batch_size * seq_len * (
        # FFNのFLOPs (簡易的な見積もり)
        4 * hidden_size * hidden_size + 
        # その他の操作 (wave変換など)
        10 * hidden_size
    )
    
    # 全層のFLOPs
    all_layers_flops = num_layers * layer_flops
    
    # 合計
    total_flops = embedding_flops + all_layers_flops
    
    return total_flops

def create_directory_for_path(path: str) -> None:
    """
    指定されたファイルパスのディレクトリが存在しない場合作成します。
    
    Args:
        path: ディレクトリを作成するファイルパス
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def get_device_info() -> Dict[str, Any]:
    """
    利用可能なデバイス情報を取得します。
    
    Returns:
        デバイス情報を含む辞書
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if info['cuda_available']:
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name(info['current_device'])
        info['memory_allocated'] = torch.cuda.memory_allocated(info['current_device'])
        info['memory_reserved'] = torch.cuda.memory_reserved(info['current_device'])
    
    return info

def setup_colab_for_a100() -> None:
    """
    Google Colabの環境でA100 GPUを使用するための最適な設定を行います。
    
    実装の詳細:
        - GPUメモリ使用の最適化
        - JITコンパイル設定
        - データローダーワーカー設定
    """
    # Google Colabでの設定
    try:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True  # 入力サイズが一定の場合はBenchmarkモードでさらに高速化
        
        # 必要に応じてDataLoaderのワーカー数を調整
        import multiprocessing as mp
        torch.set_num_threads(mp.cpu_count() - 1)
        
        print("Google Colab環境のA100 GPU用に最適化設定を適用しました")
    except Exception as e:
        print(f"最適化設定の適用中にエラーが発生しました: {e}")

def log_to_tensorboard(writer, values: Dict[str, float], step: int, prefix: str = "") -> None:
    """
    複数のメトリクスをTensorBoardに一度に記録します。
    
    Args:
        writer: TensorBoardのSummaryWriterインスタンス
        values: 記録する値の辞書 {名前: 値}
        step: 現在のステップ（x軸）
        prefix: メトリクス名のプレフィックス
    """
    for name, value in values.items():
        metric_name = f"{prefix}/{name}" if prefix else name
        writer.add_scalar(metric_name, value, step)

def enable_amp_training(model: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[torch.cuda.amp.GradScaler, Any]:
    """
    混合精度トレーニングのセットアップを行います。
    
    Args:
        model: トレーニング対象のモデル
        optimizer: オプティマイザ
        
    Returns:
        Tuple: (GradScaler, model) AMP
    """
    scaler = torch.cuda.amp.GradScaler()
    model = torch.nn.parallel.DistributedDataParallel(model)
    return scaler, model


