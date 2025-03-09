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
        dropout_prob: ドロップアウト率
        use_rope: RoTary Position Embeddingを使うかどうか
        use_wavelet: Wavelet変換を使用するかどうか (True/False)
        wavelet_name: PyWaveletsで使用するWaveletの名称 (例: 'haar', 'db1', 'db2'など)
        norm_scheme: 'pre'または'post'のLayerNorm方式を選択
        activation: 活性化関数の選択
        complex_init_scale: 複素数初期化のスケール
    """
    def __init__(
        self,
        hidden_size: int = 1024,
        num_layers: int = 6,
        vocab_size: Optional[int] = None,  # トークナイザーから取得する場合はNone
        max_seq_len: int = 512,
        dropout_prob: float = 0.2,
        use_rope: bool = True,
        use_wavelet: bool = False,
        wavelet_name: Optional[str] = None,
        norm_scheme: str = "post",  # 追加: 'pre'または'post'のLayerNorm方式を選択
        activation: str = "gelu",   # 追加: 活性化関数の選択
        complex_init_scale: float = 0.02,  # 追加: 複素数初期化のスケール
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dropout_prob = dropout_prob
        self.use_rope = use_rope
        self.use_wavelet = use_wavelet
        self.wavelet_name = wavelet_name
        self.tokenizer = None  # トークナイザーを後から設定可能に
        # self.norm_scheme = norm_scheme  # 追加: Pre-LN vs Post-LN
        self.activation = activation  # 追加: 活性化関数
        self.complex_init_scale = complex_init_scale  # 追加: 複素数初期化スケール
    
    @property
    def vocab_size(self) -> int:
        """
        語彙サイズを取得。トークナイザーが設定されている場合はそこから取得、
        なければ初期化時に設定された値を使用
        """
        if self.tokenizer is not None:
            # transformers AutoTokenizerのサポートを追加
            if hasattr(self.tokenizer, 'vocab'):
                return len(self.tokenizer.vocab)
            elif hasattr(self.tokenizer, 'vocab_size'):
                return self.tokenizer.vocab_size
            elif hasattr(self.tokenizer, 'sp') and hasattr(self.tokenizer.sp, 'get_piece_size'):
                return self.tokenizer.sp.get_piece_size()
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
        トレーニング関連の設定を保持するクラス。
        学習率、バッチサイズ、エポック数などを指定。
    """
    def __init__(
        self,
        learning_rate: float = 1e-5,  # 1e-4だと数値が不安定になりロスにnanがでる
        batch_size: int = 32,
        mlm_epochs: int = 1,
        mlm_probability: float = 0.2,
        diffusion_epochs: int = 0,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        accumulation_steps: int = 1,  # 勾配累積ステップ数
        max_steps: Optional[int] = None,  # 最大ステップ数（Noneの場合はエポック数で制御）
        # 以下に新しい設定項目を追加
        use_amp: bool = True,  # 混合精度トレーニング
        use_gradient_checkpointing: bool = True,  # 勾配チェックポイント
        clip_grad_norm: Optional[float] = True,  # 勾配クリッピング
        clip_value: float = 1.0,  # 追加: 勾配クリッピング値
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mlm_epochs = mlm_epochs
        self.mlm_probability = mlm_probability
        self.diffusion_epochs = diffusion_epochs
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.accumulation_steps = accumulation_steps
        self.max_steps = max_steps
        # メモリ効率的な学習の設定
        self.use_amp = use_amp
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.gradient_accumulation_steps = accumulation_steps  # 互換性のため
        self.clip_grad_norm = clip_grad_norm
        self.clip_value = clip_value  # 追加: 勾配クリッピング値


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
        tokenizer_name: str = "cl-tohoku/bert-base-japanese-whole-word-masking", # 修正: 正しいリポジトリ名
        tokenizer_file: str = "tokenizer_model.json" # 追加: 実際のファイル名を指定
    ) -> None:
        self.base_dir = base_dir
        self.data_dir = os.path.join(self.base_dir, "data")
        self.checkpoint_dir = os.path.join(self.base_dir, "checkpoints")
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.tokenizer_name = tokenizer_name
        self.tokenizer_file = tokenizer_file  # 追加: ファイル名を保存
    
    @property
    def dataset_dir(self) -> str:
        """データセットの保存ディレクトリパスを返します"""
        if self.dataset_subset:
            return os.path.join(self.data_dir, self.dataset_name, self.dataset_subset)
        return os.path.join(self.data_dir, self.dataset_name)
        
    @property
    def tokenizer_path(self) -> str:
        """トークナイザーモデルのパスを返します"""
        return os.path.join(self.checkpoint_dir, "tokenizers", self.tokenizer_file)
        
    @property
    def model_save_dir(self) -> str:
        """モデル保存ディレクトリを返します"""
        return self.checkpoint_dir
        
    @property
    def tensorboard_log_dir(self) -> str:
        """TensorBoardのログディレクトリを返します"""
        return os.path.join(self.log_dir, "tensorboard")

# EasyConfigクラスを追加してファイル名からYAML/JSON/INI形式の設定を読み込める機能を提供
class EasyConfig:
    """シンプルな設定管理クラス"""
    
    def __init__(self, **kwargs):
        """キーワード引数から属性を初期化"""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_file(cls, file_path):
        """ファイルから設定を読み込む"""
        import yaml
        import json
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {file_path}")
        
        # ファイル拡張子に基づいて適切なローダーを選択
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if ext in ['.yml', '.yaml']:
                    config_dict = yaml.safe_load(f)
                elif ext == '.json':
                    config_dict = json.load(f)
                else:
                    # デフォルトはJSON形式として読み込み
                    config_dict = json.load(f)
                    
            return cls(**config_dict)
            
        except Exception as e:
            raise ValueError(f"設定ファイルの読み込み中にエラーが発生しました: {e}")
    
    def to_dict(self):
        """設定を辞書形式に変換"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, file_path):
        """設定をファイルに保存"""
        import yaml
        import json
        
        config_dict = self.to_dict()
        ext = os.path.splitext(file_path)[1].lower()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if ext in ['.yml', '.yaml']:
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                json.dump(config_dict, f, indent=2)

def get_config(config_file=None):
    """設定を読み込む便利な関数
    config_fileが指定されていればそこから読み込み、なければデフォルト値を使用
    """
    if config_file and os.path.exists(config_file):
        return EasyConfig.from_file(config_file)
    
    # デフォルト設定
    return EasyConfig(
        # デフォルト値
        model_name="cl-tohoku/bert-base-japanese-whole-word-masking",
        output_dir="/tmp/wave_network_output",
        cache_dir="./.cache",
        dataset_name="singletongue/wikipedia-utils",
        dataset_subset="corpus-jawiki-20230403-filtered-large",
        language="ja",
        
        # モデル構成パラメータ
        hidden_size=768,
        ffn_dim=3072,  # 通常はhidden_sizeの4倍
        num_layers=3,
        max_seq_len=512,
        dropout_prob=0.1,
        
        # トレーニングパラメータ
        batch_size=32,
        learning_rate=1e-5,
        num_epochs=3,
        warmup_steps=500,
        weight_decay=0.01,
        num_samples=100  # 埋め込み分析用
    )
