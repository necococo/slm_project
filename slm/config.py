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
        hidden_size: int = 1024,
        num_layers: int = 3,
        vocab_size: Optional[int] = None,  # トークナイザーから取得する場合はNone
        max_seq_len: int = 2048,
        use_rope: bool = True,
        use_wavelet: bool = False,
        wavelet_name: Optional[str] = None
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
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
        self.tokenizer = tokenizer
        
class TrainingConfig:
    """
    How:
        トレーニング関連の設定を保持するクラス。
        学習率、バッチサイズ、エポック数などを指定。
    """
    def __init__(
        self,
        learning_rate: float = 1e-5, # 1e-4だと数値が不安定になりロスにnanがでる
        batch_size: int = 32,
        mlm_epochs: int = 3,
        diffusion_epochs: int = 0,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        # 以下に新しい設定項目を追加
        use_amp: bool = True,  # 混合精度トレーニング
        use_gradient_checkpointing: bool = True,  # 勾配チェックポイント
        gradient_accumulation_steps: int = 1,  # 勾配累積ステップ数
        clip_grad_norm: Optional[float] = True,  # 勾配クリッピング
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mlm_epochs = mlm_epochs
        self.diffusion_epochs = diffusion_epochs
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        # メモリ効率的な学習の設定
        self.use_amp = use_amp
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.clip_grad_norm = clip_grad_norm


class PathsConfig:
    """
    How:
        データや重み、その他の出力を保存するパスを管理します。

    Attributes:
        data_dir: データセットを保存するベースディレクトリ
        checkpoint_dir: 学習済みモデル(重みファイル)を保存するディレクトリ
        log_dir: TensorBoardログや学習進捗を保存するディレクトリ
        dataset_name: 使用するデータセット名（例："shunk031/JGLUE", "wikitext"）
        dataset_subset: データセットのサブセット（例："JSQuAD", "wikitext-103-raw-v1"）
    """
    def __init__(
        self,
        base_dir: str = "/content/drive/MyDrive/slm",
        dataset_name: str = "singletongue/wikipedia-utils",
        dataset_subset: Optional[str] = "corpus-jawiki-20230403-filtered-large",
        # tokenizer_name: str = "NovelAI/nerdstash-tokenizer-v2" 
        tokenizer_name: str = "cl-tohoku/bert-base-japanese-whole-word-masking"
        
    ) -> None:
        self.base_dir = base_dir
        self.data_dir = os.path.join(self.base_dir, "data")
        self.checkpoint_dir = os.path.join(self.base_dir, "checkpoints")
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.tokenizer_name = tokenizer_name
        
    @property
    def dataset_dir(self) -> str:
        """データセットの保存ディレクトリパスを返します"""
        if self.dataset_subset:
            return os.path.join(self.data_dir, self.dataset_name, self.dataset_subset)
        return os.path.join(self.data_dir, self.dataset_name)
        
    @property
    def tokenizer_path(self) -> str:
        """トークナイザーモデルのパスを返します"""
        return os.path.join(self.checkpoint_dir, "tokenizers", "tokenizer.model")
        
    @property
    def model_save_dir(self) -> str:
        """モデル保存ディレクトリを返します"""
        return self.checkpoint_dir
        
    @property
    def tensorboard_log_dir(self) -> str:
        """TensorBoardのログディレクトリを返します"""
        return os.path.join(self.log_dir, "tensorboard")
