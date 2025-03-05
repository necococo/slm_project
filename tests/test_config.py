# test_config.py
from slm.config import Config

def test_config_values():
    config = Config()
    # 各パラメータが正の数になっているか確認
    assert config.vocab_size > 0, "vocab_size should be > 0"
    assert config.hidden_size in [512, 1024], "hidden_size should be 512 or 1024"
    assert config.num_layers >= 1, "num_layers should be >= 1"
    assert config.batch_size > 0, "batch_size should be > 0"
    assert config.max_seq_length > 0, "max_seq_length should be > 0"