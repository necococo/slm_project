"""
比較用の標準Transformerモデルの実装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, Union, Any

class TransformerSelfAttention(nn.Module):
    """
    標準的なマルチヘッドセルフアテンション実装
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int = 8, 
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            num_heads: アテンションヘッドの数
            dropout: ドロップアウト率
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_sizeがnum_headsで割り切れません"
        
        # 線形射影層
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
        # スケーリング係数
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        アテンションのために形状を変換
        [batch_size, seq_len, hidden_size] -> 
        [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        セルフアテンションのフォワードパス
        
        Args:
            x: 入力テンソル [batch_size, seq_len, hidden_size]
            attention_mask: アテンションマスク [batch_size, 1, 1, seq_len]
            
        Returns:
            出力テンソル [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # クエリ・キー・バリューの射影
        q = self.transpose_for_scores(self.q_proj(x))  # [B, H, S, D/H]
        k = self.transpose_for_scores(self.k_proj(x))  # [B, H, S, D/H]
        v = self.transpose_for_scores(self.v_proj(x))  # [B, H, S, D/H]
        
        # アテンションスコア計算
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, S, S]
        
        # アテンションマスクの適用（必要な場合）
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # ソフトマックス
        attention_probs = F.softmax(attention_scores, dim=-1)  # [B, H, S, S]
        attention_probs = self.dropout(attention_probs)
        
        # 値との積
        context = torch.matmul(attention_probs, v)  # [B, H, S, D/H]
        
        # 形状を元に戻す
        context = context.permute(0, 2, 1, 3).contiguous()  # [B, S, H, D/H]
        context = context.view(batch_size, seq_len, self.hidden_size)  # [B, S, D]
        
        # 出力射影
        output = self.output_proj(context)  # [B, S, D]
        
        return output


class TransformerEncoderLayer(nn.Module):
    """
    標準的なTransformerエンコーダーレイヤー
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int = 8, 
        intermediate_size: int = 3072, 
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            num_heads: アテンションヘッドの数
            intermediate_size: FFNの中間層サイズ
            dropout: ドロップアウト率
        """
        super().__init__()
        
        # セルフアテンション
        self.attention = TransformerSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # フィードフォワードネットワーク
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        
        # レイヤーノーマリゼーション
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # ドロップアウト
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
            attention_mask: アテンションマスク [batch_size, 1, 1, seq_len]
            
        Returns:
            出力テンソル [batch_size, seq_len, hidden_size]
        """
        # セルフアテンション
        attn_output = self.attention(self.norm1(x), attention_mask)
        x = x + self.dropout(attn_output)
        
        # フィードフォワードネットワーク
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        
        return x


class TransformerModel(nn.Module):
    """
    標準的なTransformerモデル実装
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
        self.intermediate_size = getattr(config, "intermediate_size", config.hidden_size * 4)
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
        
        # レイヤーノーマリゼーションとドロップアウト
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # エンコーダーレイヤー
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                intermediate_size=self.intermediate_size,
                dropout=config.dropout
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
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def create_position_ids(self, input_ids):
        """
        位置IDを生成
        """
        mask = input_ids.ne(self.pad_token_id).long()
        position_ids = torch.cumsum(mask, dim=1) * mask
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
        
        # アテンションマスク処理
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)
        
        # アテンションマスクを拡張
        extended_attention_mask = (
            attention_mask[:, None, None, :]
            .expand(batch_size, 1, seq_len, seq_len)
            .to(dtype=torch.float32)
        )
        
        # マスクを変換（0→-10000, 1→0）
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # 位置IDの処理
        if position_ids is None:
            position_ids = self.create_position_ids(input_ids)
            
        # 埋め込みを適用
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        # 埋め込み合成
        embeddings = word_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # エンコーダーレイヤーを通過
        hidden_states = embeddings
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states, extended_attention_mask)
            
        # 最終レイヤーノーム
        hidden_states = self.layer_norm(hidden_states)
        
        # 語彙に射影
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


class TransformerForMaskedLM(nn.Module):
    """
    マスク言語モデリング（MLM）用のTransformerモデル
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = TransformerModel(config)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        モデルのフォワードパス
        
        Args:
            input_ids: 入力トークンID
            attention_mask: アテンションマスク
            position_ids: 位置ID
            labels: ターゲットラベル
            
        Returns:
            出力辞書（損失値含む）
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        # 損失計算（必要な場合）
        loss = None
        if labels is not None:
            from cut_cross_entropy import linear_cross_entropy
            
            # モデル出力を取得
            hidden_states = outputs["hidden_states"]
            logits = outputs["logits"]
            
            # 半精度に変換
            if hidden_states.dtype != torch.float16:
                hidden_states = hidden_states.half()
                
            classifier_weights = self.transformer.get_classifier_weights()
            if classifier_weights.dtype != torch.float16:
                classifier_weights = classifier_weights.half()
            
            # Cut Cross Entropy
            loss = linear_cross_entropy(
                hidden_states,
                classifier_weights,
                labels
            )
            
            outputs["loss"] = loss
        
        return outputs
