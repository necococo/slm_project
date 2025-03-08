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
        use_cut_cross_entropy: Cut Cross Entropyを使用するかどうか
    """
    def __init__(
        self,
        hidden_size: int = 768,  # より大きな値に調整（例：1024）
        num_layers: int = 6,     # レイヤー数を増やす（例：6-12）
        vocab_size: Optional[int] = None,  # トークナイザーから取得する場合はNone
        max_seq_len: int = 512,
        dropout_prob: float = 0.2,
        use_rope: bool = True,
        use_wavelet: bool = False,
        wavelet_name: Optional[str] = None,
        norm_scheme: str = "post",  # 追加: 'pre'または'post'のLayerNorm方式を選択
        activation: str = "gelu",   # 追加: 活性化関数の選択
        complex_init_scale: float = 0.02,  # 追加: 複素数初期化のスケール
        use_cut_cross_entropy: bool = True,  # Cut Cross Entropyを使用するかどうか
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
        self.use_cut_cross_entropy = use_cut_cross_entropy  # インスタンス変数として設定
    
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
        learning_rate: float = 1e-3,  # 1e-4だと数値が不安定になりロスにnanがでる
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
        
    def get_effective_learning_rate(self) -> float:
        """
        実際に使用される学習率を取得します。
        auto_adjust_learning_rateがFalseの場合、設定された学習率をそのまま返します。
        
        Returns:
            float: 使用する学習率
        """
        return self.learning_rate


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
        dataset_name = None,
        dataset_subset = None,
        tokenizer_name = None, # 修正: 正しいリポジトリ名
        tokenizer_file = None, # 追加: 実際のファイル名を指定
        language: str = "ja"  # 追加: 言語選択（'ja'または'en'）
    ) -> None:
        # 言語に基づいてデータセットとトークナイザーを設定
        if language == "en":
            # 英語用のデフォルト設定 - より大きなデータセットに変更
            self.dataset_name = "wikitext"
            # wikitext-2-v1からwikitext-103-v1に変更
            self.dataset_subset = "wikitext-103-v1"  # より長いシーケンスを含む大規模データセット
            self.tokenizer_name = "gpt2"
            self.tokenizer_file = "gpt2_tokenizer.json"
        else:
            # 日本語用（既存の設定を維持）
            self.dataset_name = "singletongue/wikipedia-utils"
            self.dataset_subset = "corpus-jawiki-20230403-filtered-large"
            self.tokenizer_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
            self.tokenizer_file = "tokenizer_model.json"

        self.base_dir = base_dir
        self.checkpoint_dir = os.path.join(self.base_dir, "checkpoints")
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.visualization_dir = os.path.join(self.log_dir, "visualizations")  # 追加: 可視化結果保存ディレクトリ
        self.language = language
        self.data_dir = os.path.join(self.base_dir, "data", self.language, self.dataset_name, self.dataset_subset)
    
    @property
    def dataset_dir(self) -> str:
        """データセットの保存ディレクトリパスを返します"""
        # 無限再帰を修正 - self.dataset_dirを呼び出すと自分自身を呼び出してしまう
        os.makedirs(self.data_dir, exist_ok=True)
        return self.data_dir
        
    @property
    def tokenizer_path(self) -> str:
        """トークナイザーモデルのパスを返します"""
        tknr_dir = os.path.join(self.checkpoint_dir, "tokenizers", self.tokenizer_name)
        os.makedirs(tknr_dir, exist_ok=True)
        return os.path.join(tknr_dir, self.tokenizer_file)
        
    @property
    def model_save_dir(self) -> str:
        """モデル保存ディレクトリを返します"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        return self.checkpoint_dir
        
    @property
    def tensorboard_log_dir(self) -> str:
        """TensorBoardのログディレクトリを作って返します"""
        tb_log_dir = os.path.join(self.log_dir, "tensorboard")
        os.makedirs(tb_log_dir, exist_ok=True)
        return tb_log_dir
        
    @property
    def visualization_path(self) -> str:
        """可視化結果の保存パスを返します"""
        os.makedirs(self.visualization_dir, exist_ok=True)
        return self.visualization_dir
        
    @staticmethod
    def safe_index(idx):
        """
        NumPy整数型やその他の特殊な整数型をPythonネイティブのint型に変換します。
        これによりdatasetsライブラリなどでの型エラーを防止します。
        
        Args:
            idx: 変換する対象のインデックス値
            
        Returns:
            int: Pythonネイティブのint型に変換された値
        """
        if hasattr(idx, 'item'):  # NumPyの数値型はitem()メソッドを持っている
            return idx.item()
        return int(idx)  # その他の場合は通常のint変換を試みる
        
    @staticmethod
    def safe_dataset_access(dataset, idx, key=None):
        """
        データセットへの安全なアクセスを提供するヘルパーメソッド。
        NumPy型のインデックスを適切に処理し、オプションでデータの特定のフィールドを取得します。
        
        Args:
            dataset: アクセス対象のデータセット
            idx: アクセスするインデックス（int, numpy.int64などの型を受け付ける）
            key: 取得するデータのキー（例：'input_ids'）、Noneの場合は全データを返す
            
        Returns:
            指定したデータセットの項目またはその特定フィールド
        """
        safe_idx = PathsConfig.safe_index(idx)
        item = dataset[safe_idx]
        if key is not None:
            return item[key]
        return item
