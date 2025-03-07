"""
Wavelet + SVD + Linear Attention を組み合わせた
計算効率の良い言語モデル実装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, Union, Any

from ..models.components.wavelet_layer import WaveletLayer, MultiResolutionWaveletLayer
from ..models.components.svd_attention import SVDAttention, HybridSVDAttention
from ..models.components.linear_attention import LinearAttention, FFTAttention

class WaveletEncoderLayer(nn.Module):
    """
    Wavelet変換ベースのエンコーダーレイヤー
    
    計算効率の良いアテンション機構と、マルチスケール
    特徴抽出を組み合わせた設計
    """
    def __init__(
        self, 
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_type: str = "svd",  # "svd", "linear", "fft"
        wavelet_families: Optional[list] = None,
        svd_rank: int = 64,
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            num_heads: アテンションヘッドの数
            dropout: ドロップアウト率
            attention_type: 使用するアテンションのタイプ
            wavelet_families: 使用するウェーブレットファミリー
            svd_rank: SVDアテンションのランク
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        # アテンションモジュール選択
        if attention_type == "svd":
            self.attention = SVDAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                rank=svd_rank,
                dropout=dropout
            )
        elif attention_type == "linear":
            self.attention = LinearAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            )
        elif attention_type == "fft":
            self.attention = FFTAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            raise ValueError(f"不明なアテンションタイプ: {attention_type}")
        
        # Wavelet変換レイヤー
        if wavelet_families:
            self.wavelet = MultiResolutionWaveletLayer(
                hidden_size=hidden_size,
                wavelet_families=wavelet_families,
                dropout=dropout
            )
        else:
            self.wavelet = WaveletLayer(
                hidden_size=hidden_size,
                dropout=dropout
            )
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        エンコーダーレイヤーのフォワードパス
        
        Args:
            x: 入力テンソル [batch_size, seq_len, hidden_size]
            attention_mask: アテンションマスク（オプション）
            
        Returns:
            処理済みテンソル [batch_size, seq_len, hidden_size]
        """
        # Self-Attention部分
        attn_output = self.attention(self.norm1(x), attention_mask)
        x = x + self.dropout(attn_output)
        
        # Wavelet変換
        wavelet_output = self.wavelet(self.norm2(x))
        x = x + wavelet_output
        
        # Feed-Forward Network
        ffn_output = self.ffn(self.norm3(x))
        x = x + ffn_output
        
        return x


class WaveletModel(nn.Module):
    """
    Wavelet + SVD + Linear Attentionを組み合わせた
    計算効率の良い言語モデル
    """
    def __init__(
        self,
        config: Any
    ):
        """
        Args:
            config: モデル設定オブジェクト
        """
        super().__init__()
        self.config = config
        
        # 基本設定
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.pad_token_id = getattr(config, "pad_token_id", 0)
        
        # 埋め込み層
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=self.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        
        # レイヤー正規化とドロップアウト
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Wavelet Encoderレイヤー
        self.encoder_layers = nn.ModuleList([
            WaveletEncoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                dropout=config.dropout,
                attention_type=config.attention_type,
                wavelet_families=config.wavelet_families if hasattr(config, "wavelet_families") else None,
                svd_rank=config.svd_rank if hasattr(config, "svd_rank") else 64
            ) 
            for _ in range(config.num_layers)
        ])
        
        # 出力層
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        
        # 重みの初期化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """
        モデルの重みを初期化
        """
        if isinstance(module, nn.Linear):
            # 線形層は範囲を抑えた初期化
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 埋め込み層も同様
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # レイヤーノームはバイアス0、重み1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def create_position_ids(self, input_ids):
        """
        位置IDを生成
        """
        # パディングトークンをマスク
        mask = input_ids.ne(self.pad_token_id).long()
        
        # 位置ID生成（パディングトークンは0）
        position_ids = torch.cumsum(mask, dim=1) * mask
        
        # 境界チェック
        return torch.clamp(position_ids, 0, self.max_position_embeddings - 1)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        モデルのフォワードパス
        
        Args:
            input_ids: 入力トークンID [batch_size, seq_len]
            attention_mask: アテンションマスク [batch_size, seq_len]
            position_ids: 位置ID [batch_size, seq_len]
            
        Returns:
            出力辞書 {"embeddings": hidden_states, "logits": logits}
        """
        batch_size, seq_len = input_ids.shape
        
        # アテンションマスク処理（なければ作成）
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)
        
        # アテンションマスクを拡張（後でブロードキャストしやすくする）
        extended_attention_mask = (
            attention_mask[:, None, None, :]
            .expand(batch_size, 1, seq_len, seq_len)
            .to(dtype=torch.float32)
        )
        
        # マスクの変換（0→-10000, 1→0）
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # 位置IDの処理（なければ作成）
        if position_ids is None:
            position_ids = self.create_position_ids(input_ids)
            
        # 埋め込みレイヤーを適用
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        # 埋め込み合成
        embeddings = word_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # エンコーダーレイヤー通過
        hidden_states = embeddings
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states, extended_attention_mask)
            
        # 最終レイヤーノーム
        hidden_states = self.layer_norm(hidden_states)
        
        # 分類器（語彙への射影）
        logits = self.classifier(hidden_states)
        
        return {
            "hidden_states": hidden_states,
            "logits": logits
        }
    
    def get_classifier_weights(self) -> torch.Tensor:
        """
        分類器の重みを取得（Cut Cross Entropy用）
        """
        return self.classifier.weight


class WaveletModelForMaskedLM(nn.Module):
    """
    マスク言語モデリング（MLM）用のWaveletモデルラッパー
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wavelet_model = WaveletModel(config)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        マスク言語モデルのフォワードパス
        
        Args:
            input_ids: 入力トークンID
            attention_mask: アテンションマスク
            position_ids: 位置ID
            labels: ターゲットラベル
            
        Returns:
            出力辞書（損失値含む）
        """
        outputs = self.wavelet_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        # 損失計算（必要な場合）
        loss = None
        if labels is not None:
            # Cut Cross Entropyに互換性を持たせる
            from cut_cross_entropy import linear_cross_entropy
            
            # モデル出力を取得
            hidden_states = outputs["hidden_states"]
            logits = outputs["logits"]
            
            # 半精度に変換(linear_cross_entropyの要件)
            if hidden_states.dtype != torch.float16:
                hidden_states = hidden_states.half()
                
            classifier_weights = self.wavelet_model.get_classifier_weights()
            if classifier_weights.dtype != torch.float16:
                classifier_weights = classifier_weights.half()
            
            # Cut Cross Entropyでロス計算
            loss = linear_cross_entropy(
                hidden_states,
                classifier_weights,
                labels
            )
            
            outputs["loss"] = loss
        
        return outputs
