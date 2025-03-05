# slm/config.py
# Why not: 設定をファイルごとに分散させると保守が難しくなるため、集中管理する。

from typing import Optional
import os

class ModelConfig:
    """
    How:
        Wave Network + RoPE + Wavelet変換のモデル構成を定義します。
        波形表現の次元数やレイヤ数、語彙サイズなどをここで指定します。

    Attributes:
        hidden_size: モデル内部の隠れ次元 (wave表現における1つの軸の長さ)
        num_layers: WaveBlock + RoPEレイヤの段数
        vocab_size: 語彙サイズ（トークナイザーから自動取得も可能）
        max_seq_len: 最大シーケンス長
        use_wavelet: Wavelet変換を使用するかどうか (True/False)
        wavelet_name: PyWaveletsで使用するWaveletの名称 (例: 'haar', 'db1', 'db2'など)
    """
    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 3,
        vocab_size: Optional[int] = None,  # トークナイザーから取得する場合はNone
        max_seq_len: int = 1024,
        use_wavelet: bool = False,
        wavelet_name: Optional[str] = None
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.use_wavelet = use_wavelet
        self.wavelet_name = wavelet_name
        self.tokenizer = None  # トークナイザーを後から設定可能に
    
    @property
    def vocab_size(self) -> int:
        """
        語彙サイズを取得。トークナイザーが設定されている場合はそこから取得、
        なければ初期化時に設定された値を使用
        """
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, 'sp') and hasattr(self.tokenizer.sp, 'get_piece_size'):
                return self.tokenizer.sp.get_piece_size()
            elif hasattr(self.tokenizer, 'vocab_size'):
                return self.tokenizer.vocab_size
            else:
                raise AttributeError("トークナイザーからvocab_sizeを取得できません")
        
        if self._vocab_size is None:
            raise ValueError("vocab_sizeが設定されておらず、トークナイザーも設定されていません")
        
        return self._vocab_size
    
    def set_tokenizer(self, tokenizer) -> None:
        """
        トークナイザーを設定し、語彙サイズを自動的に取得できるようにする
        """
        self.tokenizer = tokenizer

        
class TrainingConfig:
    """
    How:
        学習時のハイパーパラメータをまとめるクラスです。

    Attributes:
        batch_size: バッチサイズ
        accumulation_steps: Gradient Accumulationのステップ数
        learning_rate: 学習率
        max_steps: 学習の最大ステップ
        mlm_mask_ratio: MLMマスク率
    """
    def __init__(
        self,
        batch_size: int = 8,
        accumulation_steps: int = 2,
        learning_rate: float = 1e-4,
        max_steps: int = 10000,
        mlm_mask_ratio: float = 0.15
    ) -> None:
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.mlm_mask_ratio = mlm_mask_ratio


class PathsConfig:
    """
    How:
        データや重み、その他の出力を保存するパスを管理します。

    Attributes:
        data_dir: データセットを保存するベースディレクトリ
        model_dir: 学習済みモデル(重みファイル)を保存するディレクトリ
        dataset_name: 使用するデータセット名（例："shunk031/JGLUE", "wikitext"）
        dataset_subset: データセットのサブセット（例："JSQuAD", "wikitext-103-raw-v1"）
    """
    def __init__(
        self,
        data_dir: str = "/content/drive/MyDrive/data",
        model_dir: str = "/content/drive/MyDrive/models",
        dataset_name: str = "shunk031/JGLUE",
        dataset_subset: str = "JSQuAD",
        tokenizer_name: str = "spm"  # または "nerdstash" など
    ) -> None:
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.tokenizer_name = tokenizer_name
        
    @property
    def dataset_dir(self) -> str:
        """データセットの保存ディレクトリパスを返します"""
        return os.path.join(self.data_dir, self.dataset_name, self.dataset_subset)
        
    @property
    def tokenizer_path(self) -> str:
        """トークナイザーモデルのパスを返します"""
        return os.path.join(self.model_dir, "tokenizers", "tokenizer.model")
        
    @property
    def model_save_dir(self) -> str:
        """モデル保存ディレクトリを返します"""
        return os.path.join(self.model_dir, "checkpoints")
