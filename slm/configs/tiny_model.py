"""
Wave Network の軽量モデル設定
SwiGLUの効率性を活かした小さい設計
"""

from slm.config import ModelConfig, TrainingConfig

def get_tiny_config() -> ModelConfig:
    """
    SwiGLUを活用した小型Wave Networkモデルの設定
    
    Returns:
        小型モデル用のModelConfig
    """
    config = ModelConfig(
        # 通常の1/2のサイズ
        hidden_size=384,  
        # レイヤー数も削減
        num_layers=2,
        # 長いシーケンスも扱えるように
        max_seq_len=1024,
        # SwiGLUを効果的に使うので少し大きめのdropoutで過学習防止
        dropout_prob=0.2,
        # 位置符号化は必須
        use_rope=True
    )
    return config

def get_tiny_training_config() -> TrainingConfig:
    """
    小型モデル用のトレーニング設定
    
    Returns:
        小型モデル向けTrainingConfig
    """
    config = TrainingConfig(
        # 小さいモデルなので学習率少し上げる
        learning_rate=2e-5,
        # バッチサイズ大きめ
        batch_size=192,
        # エポック数増加
        mlm_epochs=5,
        # マスク確率
        mlm_probability=0.25,
        # 小さいモデルなのでウォームアップは短く
        warmup_steps=200,
        # メモリ効率良いので勾配累積不要
        accumulation_steps=1,
    )
    return config
