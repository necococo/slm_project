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

def load_checkpoint(checkpoint_path: str, model, optimizer=None, device="cpu"):
    """
    保存されたチェックポイントからモデルとオプティマイザの状態を読み込む関数
    
    Args:
        checkpoint_path (str): 保存されたチェックポイントのパス。
        model (torch.nn.Module): 読み込み先のモデル。
        optimizer (torch.optim.Optimizer, optional): オプティマイザ（存在する場合）。
        device (str or torch.device, optional): モデルを配置するデバイス。
    
    Returns:
        int: チェックポイントに記録されたエポック番号（なければ0）。
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    print(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}")
    return epoch

def get_model_size(model: nn.Module) -> int:
    """
    モデルのパラメータ数（トレーニング可能な重みの数）を計算します。
    
    Args:
        model: サイズを計算するモデル
        
    Returns:
        パラメータの総数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_flops_per_batch(model: nn.Module, input_shape: Tuple[int, ...], detailed: bool = False) -> Union[float, Dict[str, float]]:
    """
    モデルの1バッチあたりの概算FLOPsを計算します。
    
    実装の詳細:
        かなり大雑把な見積もりで、実際のFLOPsは実装によって大きく変わります。
        主要な行列積演算や活性化関数の計算量を基に推定値を出します。
        
    Args:
        model: FLOPsを計算するモデル
        input_shape: 入力テンソルの形状。通常は(batch_size, seq_len)
        detailed: Trueの場合は各レイヤー別の計算量も返す
        
    Returns:
        推定されるFLOPs数
    """
    # WaveHierarchicalLM向けの簡易FLOPs計算
    batch_size, seq_len = input_shape
    hidden_size = model.config.hidden_size if hasattr(model, 'config') else 256
    num_layers = model.config.num_layers if hasattr(model, 'config') else 3
    vocab_size = model.config.vocab_size if hasattr(model, 'config') else 30000
    
    # 各コンポーネントのFLOPs見積もり
    embedding_flops = batch_size * seq_len * hidden_size  # Embedding lookup
    
    # Wave表現への変換
    wave_conversion_flops = batch_size * seq_len * hidden_size * 5  # G計算+三角関数+乗算
    
    # WaveBlock計算 (各レイヤー)
    single_layer_flops = batch_size * seq_len * (
        # 線形変換x2
        2 * hidden_size * hidden_size * 2 * 2 +
        # 加算と波形変換
        hidden_size * 10
    )
    
    # RoPE適用（使用時）
    rope_flops = 0
    if hasattr(model, 'use_rope') and model.use_rope:
        rope_flops = batch_size * seq_len * hidden_size * 6  # cos/sin乗算
    
    # 出力層
    output_flops = batch_size * seq_len * hidden_size * 2 * vocab_size
    
    # 合計
    total_flops = (
        embedding_flops + 
        wave_conversion_flops + 
        single_layer_flops * num_layers +
        rope_flops * num_layers +
        output_flops
    )
    
    if detailed:
        return {
            'embedding': embedding_flops,
            'wave_conversion': wave_conversion_flops,
            'wave_blocks': single_layer_flops * num_layers,
            'rope': rope_flops * num_layers,
            'output': output_flops,
            'total': total_flops
        }
    
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


