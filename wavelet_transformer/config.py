"""
Wavelet TransformerとベースラインTransformerの設定クラス
A100 GPU上で24時間の制約内で動作するための最適化された設定
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any

@dataclass
class BaseConfig:
    """
    モデル・学習設定の基底クラス
    両方のモデルタイプに共通するパラメータを定義します
    """
    # モデル基本設定
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 12
    vocab_size: int = 30522  # BERTの語彙サイズ（デフォルト）
    max_position_embeddings: int = 512
    dropout: float = 0.1
    
    # 学習設定
    batch_size: int = 32
    eval_batch_size: int = 64  # 評価時のバッチサイズ
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 5
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1  # 総ステップ数に対する比率
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    scheduler_step_by: str = "epoch"  # "epoch" または "batch"
    
    # 評価・保存設定
    save_every: int = 1
    early_stop_patience: int = 3
    eval_final_model: bool = True
    
    # データ処理設定
    mlm_probability: float = 0.15  # マスク言語モデリングのマスク率
    max_seq_length: int = 512
    num_workers: int = 4
    
    # 最適化設定
    use_amp: bool = True  # 自動混合精度
    fp16: bool = True  # 半精度学習
    bf16: bool = False  # bfloat16（A100ではより高速な場合がある）
    
    # パス設定
    output_dir: str = "/content/output"
    log_dir: str = "/content/output/logs"
    
    # デバッグ設定
    debug: bool = False  # デバッグモード
    debug_samples: int = 1000  # デバッグ時のサンプル数

@dataclass
class WaveletModelConfig(BaseConfig):
    """
    Waveletモデル専用の設定
    Wavelet特有のパラメータを追加します
    """
    # Wavelet特有の設定
    wavelet_families: List[str] = field(default_factory=lambda: ["db2", "haar", "sym2"])
    decomposition_level: int = 3
    attention_type: str = "svd"  # "svd", "linear", "fft"
    svd_rank: int = 64
    use_multi_resolution: bool = True
    
    # モデル構造最適化
    num_layers: int = 8  # より多くの階層で情報を抽出
    hidden_size: int = 512  # 適度なサイズ
    
    def __post_init__(self):
        """設定の整合性チェックと調整"""
        # SVDランクは隠れ層サイズの1/4を超えないように
        self.svd_rank = min(self.svd_rank, self.hidden_size // 4)
        
        # 効率化のためのバッチサイズ調整（SVD計算のため）
        if self.attention_type == "svd" and self.batch_size > 24:
            print(f"注意: SVDアテンションではバッチサイズを小さくすることを推奨 ({self.batch_size} -> 24)")
            self.batch_size = 24
            
        # 更に出力ディレクトリを専用化
        self.output_dir = f"{self.output_dir}/wavelet_model_{self.attention_type}"
        self.log_dir = f"{self.log_dir}/wavelet_model_{self.attention_type}"

@dataclass
class TransformerModelConfig(BaseConfig):
    """
    標準的なTransformerモデルの設定
    BERTと同等の設定をベースにした比較用モデル
    """
    # Transformer特有の設定
    intermediate_size: int = 3072  # FFN中間層のサイズ
    hidden_act: str = "gelu"  # 活性化関数
    layer_norm_eps: float = 1e-12
    
    # アーキテクチャ設定
    num_layers: int = 6  # シンプルな比較のため層数を少なく
    hidden_size: int = 768
    
    def __post_init__(self):
        """設定の整合性チェックと調整"""
        # FFNサイズの調整
        if self.intermediate_size == 0:
            self.intermediate_size = self.hidden_size * 4
            
        # 出力ディレクトリの調整
        self.output_dir = f"{self.output_dir}/transformer_model"
        self.log_dir = f"{self.log_dir}/transformer_model"

def get_wavelet_config(size: str = "base") -> WaveletModelConfig:
    """
    事前定義されたWaveletモデル設定を取得
    
    Args:
        size: モデルサイズ "small", "base", "large"
        
    Returns:
        対応するWaveletModelConfig
    """
    if size == "small":
        return WaveletModelConfig(
            hidden_size=384,
            num_layers=6,
            num_heads=6,
            svd_rank=32,
            batch_size=64,
            decomposition_level=2
        )
    elif size == "base":
        return WaveletModelConfig(
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            svd_rank=64,
            batch_size=32,
            decomposition_level=3
        )
    elif size == "large":
        return WaveletModelConfig(
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            svd_rank=96,
            batch_size=16,
            decomposition_level=3,
            gradient_accumulation_steps=2
        )
    else:
        raise ValueError(f"Unknown model size: {size}")

def get_transformer_config(size: str = "base") -> TransformerModelConfig:
    """
    事前定義されたTransformerモデル設定を取得
    
    Args:
        size: モデルサイズ "small", "base", "large"
        
    Returns:
        対応するTransformerModelConfig
    """
    if size == "small":
        return TransformerModelConfig(
            hidden_size=384, 
            num_layers=6,
            num_heads=6,
            intermediate_size=1536,
            batch_size=64
        )
    elif size == "base":
        return TransformerModelConfig(
            hidden_size=768,
            num_layers=6,
            num_heads=12,
            intermediate_size=3072,
            batch_size=32
        )
    elif size == "large":
        return TransformerModelConfig(
            hidden_size=1024,
            num_layers=8,
            num_heads=16, 
            intermediate_size=4096,
            batch_size=16,
            gradient_accumulation_steps=2
        )
    else:
        raise ValueError(f"Unknown model size: {size}")
