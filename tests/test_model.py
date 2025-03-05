# slm/tests/test_model.py
# Why not: 複素数表現での順伝播がshape的に合うかなどをチェック。

import pytest
import torch
from slm.config import ModelConfig
from slm.model import WaveHierarchicalLM, SingleWaveNetworkLayer, WaveNetworkBlock

@pytest.mark.parametrize("use_wavelet", [False, True])
def test_model_forward(use_wavelet: bool) -> None:
    """
    What:
        WaveHierarchicalLMを前向き推論して、
        出力shapeが (batch, seq, vocab_size) になるか確認。
        Waveletをon/offしたときの両方をテスト。
    """
    config = ModelConfig(
        hidden_size=16,
        num_layers=2,
        vocab_size=100,
        max_seq_len=32,
        use_wavelet=use_wavelet,
        wavelet_name="haar"
    )
    model = WaveHierarchicalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))  # batch=2, seq=10
    logits = model(input_ids)
    assert logits.shape == (2, 10, config.vocab_size)


def test_single_wave_network_layer() -> None:
    """
    What:
        SingleWaveNetworkLayerの前向き推論をテストし、
        出力shapeが入力と一致するか確認。
    """
    layer = SingleWaveNetworkLayer(hidden_size=16)
    real_in = torch.randn(2, 10, 16)  # batch=2, seq=10, hidden=16
    imag_in = torch.randn(2, 10, 16)
    real_out, imag_out = layer(real_in, imag_in)
    assert real_out.shape == real_in.shape
    assert imag_out.shape == imag_in.shape


def test_wave_network_block() -> None:
    """
    What:
        WaveNetworkBlockの前向き推論をテストし、
        出力shapeが入力と一致するか確認。
    """
    block = WaveNetworkBlock(hidden_size=16, num_layers=2)
    real_in = torch.randn(2, 10, 16)  # batch=2, seq=10, hidden=16
    imag_in = torch.randn(2, 10, 16)
    real_out, imag_out = block(real_in, imag_in)
    assert real_out.shape == real_in.shape
    assert imag_out.shape == imag_in.shape

