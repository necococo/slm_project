# slm/model.py

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.rope import RoPEEncoding  # Why not: ropeは外部モジュールとして利用する
from .config import ModelConfig
from .modules.wavelayer import SingleWaveLayer, to_wave_representation


class WaveNetworkLM(nn.Module):
    """
    How:
        - Embedding => Wave表現に変換
        - (option) RoPE on Wave representation (real, imag)
        - WaveNetwork (SingleWaveLayer×N)
        - output_proj (classifier weights) は持つが forward では logitsを作らず、埋め込みだけ返す。

    Why not:
        通常CEを使わず、cut-cross-entropy の linear_cross_entropy() のみを使うので、
        モデル自体は logits を出さずに済む。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        vocab_size = config.vocab_size

        # 埋め込み
        self.embedding = nn.Embedding(vocab_size, config.hidden_size)

        # RoPE (Wave表現に対して適用)
        self.rope = RoPEEncoding(config.hidden_size, config.max_seq_len) if config.use_rope else None

        # Wave blocks
        self.layers = nn.ModuleList([
            SingleWaveLayer(config.hidden_size)
            for _ in range(config.num_layers)
        ])

        # 分類器（cut-cross-entropy用の重み）
        self.classifier = nn.Linear(config.hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        How:
            1) Embedding => (B,S,D)
            2) Wave表現に変換 => (real, imag)
            3) (option) RoPE => (real, imag)
            4) Wave layers => (B,S,D)
            5) return: (B,S,D)  (cut-cross-entropyの embedding)
        """
        x = self.embedding(input_ids)  # (B,S,D)

        # Wave表現に変換
        real, imag = to_wave_representation(x)

        # RoPE適用（オプション）
        if self.rope is not None:
            real, imag = self.rope(real, imag)  # Wave表現に対してRoPEを適用

        # SingleWaveLayerは実ベクトルを入出力するので、Wave表現からreal部分のみ取り出す、
        # または実部と虚部を結合して渡す実装に修正する必要があります
        # ここでは実装を簡略化するため、Wave表現から実ベクトルに戻します
        x = torch.sqrt(real**2 + imag**2)  # 振幅を計算

        for layer in self.layers:
            x = layer(x)  # (B,S,D)

        # 埋め込みのみを返す（cut-cross-entropyに必要）
        return x  # => embeddings (B,S,D)
    
    def get_classifier_weights(self) -> torch.Tensor:
        """
        How:
            cut-cross-entropyで linear_cross_entropy() を呼ぶ際に必要となる
            分類器の重み (V, D) を返す。
        """
        # classifier.weight shape: (V, D)
        return self.classifier.weight
