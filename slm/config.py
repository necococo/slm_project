# slm/config.py
# Why not: 設定をファイルごとに分散させると保守が難しくなるため、集中管理する。

from typing import Optional
import os

class ModelConfig:
    """
    Wiki40B日本語データセット専用のWave Network設定
    
    Attributes:
        hidden_size: モデル内部の隠れ次元
        num_layers: WaveBlock + RoPEレイヤの段数
        vocab_size: 語彙サイズ
        max_seq_len: 最大シーケンス長
        dropout_prob: ドロップアウト率
    """
    def __init__(
        self,
        hidden_size: int = 1024,
        num_layers: int = 3,
        vocab_size: Optional[int] = 32100,  # megagonlabs/t5-base-japanese-webの語彙サイズ
        max_seq_len: int = 512,
        dropout_prob: float = 0.2,
        use_rope: bool = True,
        # 以下はデフォルト値で固定し、ユーザーが変更する必要のないパラメータ
        use_wavelet: bool = True,
        wavelet_name: str = "haar",
        norm_scheme: str = "post",
        activation: str = "gelu",
        complex_init_scale: float = 0.02,
        use_bio_noise: bool = True,
        noise_std: float = 0.1,
        trainable_noise: bool = False,
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
        self.complex_init_scale = complex_init_scale
        
        # 生体ゆらぎ関連の設定
        self.use_bio_noise = use_bio_noise
        self.noise_std = noise_std
        self.trainable_noise = trainable_noise
        
        # ウェーブレット関連の設定
        self.use_wavelet = use_wavelet
        self.wavelet_name = wavelet_name
    
    @property
    def vocab_size(self) -> int:
        """
        語彙サイズを取得。トークナイザーが設定されている場合はそこから取得、
        なければ初期化時に設定された値を使用
        """
        if self.tokenizer is not None:
            try:
                # まず最も確実な方法で取得を試みる
                # 1. vocab_sizeプロパティ（最も一般的）
                if hasattr(self.tokenizer, 'vocab_size'):
                    return self.tokenizer.vocab_size
                
                # 2. vocabディクショナリ
                elif hasattr(self.tokenizer, 'vocab') and isinstance(self.tokenizer.vocab, dict):
                    return len(self.tokenizer.vocab)
                
                # 3. SPのget_piece_sizeメソッド
                elif hasattr(self.tokenizer, 'sp') and hasattr(self.tokenizer.sp, 'get_piece_size'):
                    try:
                        return self.tokenizer.sp.get_piece_size()
                    except Exception as e:
                        print(f"警告: get_piece_sizeの呼び出しに失敗しました: {e}")
                
                # 4. _tokenizerの確認 (SPEmulatorラッパー用)
                elif hasattr(self.tokenizer, '_tokenizer'):
                    if hasattr(self.tokenizer._tokenizer, 'vocab_size'):
                        return self.tokenizer._tokenizer.vocab_size
                    elif hasattr(self.tokenizer._tokenizer, 'vocab'):
                        return len(self.tokenizer._tokenizer.vocab)
                
                # 5. コード内にハードコードされた値を使用（最後の手段）
                print("警告: トークナイザーから語彙サイズを取得できません。デフォルト値を使用します。")
                return 32000  # 一般的なT5モデルの語彙サイズ
                
            except Exception as e:
                print(f"警告: トークナイザーから語彙サイズの取得に失敗しました: {e}")
                if self._vocab_size is not None:
                    print(f"初期化時に指定された語彙サイズ {self._vocab_size} を使用します")
                    return self._vocab_size
                else:
                    print("デフォルト値 32000 を使用します")
                    return 32000  # 安全のためのデフォルト値
        
        if self._vocab_size is None:
            print("警告: vocab_sizeが設定されておらず、トークナイザーも設定されていません。デフォルト値 32000 を使用します")
            return 32000  # 安全のためのデフォルト値
        
        return self._vocab_size
    
    def set_tokenizer(self, tokenizer) -> None:
        """
        トークナイザーを設定し、語彙サイズを自動的に取得できるようにする
        """
        self.tokenizer = tokenizer
        
class TrainingConfig:
    """
    Wiki40B日本語データセット専用のトレーニング設定
    
    Attributes:
        learning_rate: 学習率
        batch_size: バッチサイズ
        diffusion_epochs: 拡散学習のエポック数
    """
    def __init__(
        self,
        learning_rate: float = 2e-5,  # 1e-4だと数値が不安定になりロスにnanがでる
        batch_size: int = 32,
        mlm_epochs: int = 0,  # MLM学習をスキップ (0にするとMLM学習は実行されない)
        mlm_probability: float = 0.2,
        diffusion_epochs: int = 3,  # diffusion学習のみを実行
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
        dataset_name: 使用するデータセット名（例："NINJAL/NWJC", "shunk031/JGLUE", "wikitext"）
        dataset_subset: データセットのサブセット（例："JSQuAD", "wikitext-103-raw-v1"）
    """
    def __init__(
        self,
        base_dir: str = os.getcwd(),  # カレントディレクトリを基準
        dataset_name: str = "fujiki",  # NINJAL Web Japanese Corpus
        dataset_subset: Optional[str] = "wiki40b_ja",
        tokenizer_name: str = "megagonlabs/t5-base-japanese-web",  # 日本語Webテキスト向けに最適化されたT5ベースのトークナイザー
        output_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        run_name: Optional[str] = None
    ):
        self.base_dir = base_dir
        
        # 一意の実行名を生成（指定がない場合）
        if run_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Google Colab環境の検出（try-exceptで囲む）
            try:
                import IPython
                is_colab = 'google.colab' in str(IPython.get_ipython())
            except (ImportError, NameError):
                is_colab = False
            self.run_name = f"run_{timestamp}_{'colab' if is_colab else 'local'}"
        else:
            self.run_name = run_name
            
        # Google Driveがマウントされているか確認
        gdrive_mount_point = '/content/drive/MyDrive'
        if os.path.exists(gdrive_mount_point):
            print("Google Driveが検出されました。結果をGoogle Driveに保存します。")
            # Google Driveに結果を保存するためのディレクトリを作成
            gdrive_output = os.path.join(gdrive_mount_point, "slm_outputs", self.run_name)
            self.output_base = gdrive_output
        else:
            # ローカル実行時は指定されたbase_dirの下に保存
            self.output_base = os.path.join(self.base_dir, "runs", self.run_name)
        
        print(f"実行ID: {self.run_name}")
        print(f"出力ディレクトリ: {self.output_base}")
        
        # 各種ディレクトリのパスを設定
        self.data_dir = os.path.join(self.base_dir, "data")
        self.checkpoint_dir = os.path.join(self.output_base, "checkpoints")
        self.log_dir = os.path.join(self.output_base, "logs")
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.tokenizer_name = tokenizer_name
        self.output_dir = output_dir if output_dir is not None else self.output_base
        self.cache_dir = cache_dir if cache_dir is not None else os.path.join(self.base_dir, "cache")
        self.visualization_path = os.path.join(self.output_dir, "visualizations")
        self.logs_path = os.path.join(self.log_dir, "tensorboard")
    
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

# No code should be placed here. The config.py file should end after the PathsConfig class.
# This appears to be model initialization code that belongs in a different module.
