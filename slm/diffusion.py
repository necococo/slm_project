# slm/diffusion.py
# Why not: Large Language Diffusion Models / LlaDA論文を簡易再現する例

import torch
import torch.nn as nn
from typing import Optional

class SimpleTextDiffusion(nn.Module):
    """
    How:
        テキストのマスクノイズを注入し、拡散過程で復元させる簡易版。
        実際にはノイズスケジュールなどを工夫する必要がある。
        WaveNetworkLM互換のインターフェース: forwardでembeddingsを、
        get_classifier_weightsでclassifierを返す設計に統一。
    """

    def __init__(self, timesteps: int = 20, mask_token_id: int = 4, vocab_size: Optional[int] = None) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        # 擬似的な分類器の重み (identity matrix)
        self.classifier = None

    def forward(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        t: int
    ) -> torch.Tensor:
        """
        How:
            tステップに応じてマスク率を変化させ、モデルに通して埋め込みを得る。
            
        Returns:
            torch.Tensor: モデルからの埋め込み出力 (embeddings)
        """
        ratio = (t + 1) / self.timesteps * 0.5
        corrupted = self._mask_tokens(input_ids, ratio)
        
        # モデルを通して埋め込みを取得
        embeddings = model(corrupted)
        
        # デバイスとvocab_sizeを記録
        if self.classifier is None and self.vocab_size is None and hasattr(model, 'get_classifier_weights'):
            classifier = model.get_classifier_weights()
            self.vocab_size = classifier.size(0)
            
        return embeddings
    
    def get_classifier_weights(self) -> torch.Tensor:
        """
        How:
            分類器の重みを返します。diffusion modelの場合は
            Identity行列を返すことで、埋め込みをそのままロジットとして使用します。
            
        Returns:
            torch.Tensor: 分類器の重み（Identity行列）
            形状は(vocab_size, hidden_size)を想定
            trainer.pyで.t()で転置するため、ここでは転置しない
        """
        if self.classifier is None:
            if self.vocab_size is None:
                raise ValueError("vocab_sizeが設定されていません。forwardを一度実行するか、初期化時にvocab_sizeを指定してください。")
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
            # trainer.pyで転置するため、ここでは(vocab_size, vocab_size)の単位行列を返す
            self.classifier = torch.eye(self.vocab_size, device=device)
        return self.classifier

    def _mask_tokens(
        self,
        input_ids: torch.Tensor,
        ratio: float
    ) -> torch.Tensor:
        device = input_ids.device
        masked = input_ids.clone()
        rand = torch.rand(input_ids.shape, device=device)
        mask_pos = rand < ratio
        masked[mask_pos] = self.mask_token_id
        return masked


