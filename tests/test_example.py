# テストコードの例
import pytest
from slm.config import ModelConfig  # srcディレクトリのコードを通常どおりインポート

def test_model_config():
    config = ModelConfig(hidden_size=128)
    assert config.hidden_size == 128
    assert config.num_layers == 3  # デフォルト値のテスト
