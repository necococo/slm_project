"""
拡張版Wave Networkモデル実装
効率性と精度を高めた最新のアーキテクチャ
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple

from slm.modules.rmsnorm import RMSNorm
from slm.modules.activations import GatedMLP, SwiGLU
from slm.linear_wave_attention import LinearWaveAttention
from slm.modules.wave_network import compute_wave_representation

class EnhancedWaveRepresentation(nn.Module):
    """
    波表現の拡張実装
    
    標準的なWave表現を改善し、より安定した表現と
    より正確な位相情報の抽出を可能にする
    """
    def __init__(
        self,
        hidden_size: int,
        decomposition_mode: str = "hybrid",  # "token", "global", "hybrid"
        epsilon: float = 1e-6,
        learnable_epsilon: bool = True
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            decomposition_mode: 波表現分解モード
            epsilon: 数値安定性のための小さな値
            learnable_epsilon: epsilonを学習可能にするか
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.decomposition_mode = decomposition_mode
        self.base_epsilon = epsilon
        
        if learnable_epsilon:
            # 学習可能なepsilon（常に正値を保つためソフトプラス関数で変換）
            self.log_epsilon = nn.Parameter(torch.ones(1) * math.log(math.exp(epsilon) - 1))
        else:
            self.register_buffer('epsilon', torch.tensor([epsilon]))
            
        # 位相を捉えるための特殊な射影層
        self.phase_mapper = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 全体の波形を微調整するための係数
        self.amplitude_scale = nn.Parameter(torch.ones(1))
        self.phase_shift = nn.Parameter(torch.zeros(1))
        
    def get_epsilon(self) -> torch.Tensor:
        """学習可能なepsilonを取得（常に正値）"""
        if hasattr(self, 'log_epsilon'):
            # ソフトプラス関数: log(1 + exp(x))
            return F.softplus(self.log_epsilon)
        return self.epsilon
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        入力テンソルから拡張Wave表現（実部と虚部）を計算
        
        Args:
            x: 入力テンソル [batch_size, seq_len, hidden_size]
            
        Returns:
            (real_part, imag_part): 波表現の実部と虚部
        """
        eps = self.get_epsilon()
        
        # 数値安定性のために明示的に float32 に変換
        x_32 = x.float() if x.dtype != torch.float32 else x
        
        if self.decomposition_mode == "token":
            # トークンレベルの表現（次元方向に平均）
            real_token, imag_token = compute_wave_representation(x_32, global_mode=False, eps=eps)
            return real_token, imag_token
            
        elif self.decomposition_mode == "global":
            # 文レベルの表現（シーケンス全体で平均）
            real_global, imag_global = compute_wave_representation(x_32, global_mode=True, eps=eps)
            return real_global, imag_global
            
        else:  # "hybrid" - トークンと文レベルの両方を組み合わせる
            # トークンレベル
            real_token, imag_token = compute_wave_representation(x_32, global_mode=False, eps=eps)
            
            # 文レベル
            real_global, imag_global = compute_wave_representation(x_32, global_mode=True, eps=eps)
            
            # 位相を微調整（トークンレベルの補正用）
            phase_adjustment = self.phase_mapper(x_32)
            adjustment_amplitude = torch.sqrt(torch.sum(phase_adjustment**2, dim=-1, keepdim=True) + eps)
            
            # 組み合わせ
            real_hybrid = real_token + real_global + adjustment_amplitude * torch.cos(self.phase_shift)
            imag_hybrid = imag_token + imag_global + adjustment_amplitude * torch.sin(self.phase_shift)
            
            # 全体のスケーリング
            real_hybrid = real_hybrid * self.amplitude_scale
            imag_hybrid = imag_hybrid * self.amplitude_scale
            
            return real_hybrid, imag_hybrid


class EnhancedWaveBlock(nn.Module):
    """
    拡張版Wave Network Block
    
    標準的なWave Blockを改良し、より効率的な情報処理と
    長距離依存関係の捕捉を実現
    """
    def __init__(
        self,
        hidden_size: int,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
        use_linear_attention: bool = True
    ):
        """
        Args:
            hidden_size: 隠れ層の次元数
            ffn_ratio: フィードフォワードネットワークの拡大率
            dropout: ドロップアウト率
            use_linear_attention: 線形計算量の注意機構を使用するか
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        # 拡張波表現
        self.wave_representation = EnhancedWaveRepresentation(
            hidden_size=hidden_size,
            decomposition_mode="hybrid"
        )
        
        # 線形Wave注意機構（オプション）
        self.use_attention = use_linear_attention
        if use_linear_attention:
            self.wave_attention = LinearWaveAttention(
                hidden_size=hidden_size,
                num_heads=hidden_size // 64  # 適切なヘッド数
            )
        
        # 効率的なフィードフォワードネットワーク
        ffn_dim = int(hidden_size * ffn_ratio)
        if ffn_dim % 2 == 1:  # GatedMLPは偶数幅を想定
            ffn_dim += 1
            
        self.ffn = GatedMLP(
            in_features=hidden_size,
            hidden_ratio=ffn_ratio / 2  # GatedMLPでは半分になるので調整
        )
        
        # 規格化レイヤー
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
        # 出力プロジェクション
        self.output_proj = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        拡張Wave Blockのフォワードパス
        
        Args:
            x: 入力テンソル [batch_size, seq_len, hidden_size]
            
        Returns:
            処理された表現 [batch_size, seq_len, hidden_size]
        """
        # 残差接続のために入力を保存
        identity = x
        
        # 波動表現に変換
        real_part, imag_part = self.wave_representation(self.norm1(x))
        
        # 線形Wave注意機構（使用する場合）
        if self.use_attention:
            # 実部と虚部を結合して注意機構に通す
            wave_concat = torch.cat([real_part, imag_part], dim=-1)
            attention_output = self.wave_attention(wave_concat)
            
            # 実部と虚部を再度分離
            real_part, imag_part = attention_output.chunk(2, dim=-1)
        
        # 波動処理: 振幅と位相に基づいて新しい表現を生成
        amplitude = torch.sqrt(real_part**2 + imag_part**2 + 1e-7)
        wave_output = torch.cat([amplitude, amplitude * torch.sign(real_part)], dim=-1)
        
        # 出力射影
        wave_output = self.output_proj(wave_output)
        wave_output = self.dropout(wave_output)
        
        # 第1の残差接続
        x = x + wave_output
        
        # フィードフォワードネットワーク
        ffn_output = self.ffn(self.norm2(x))
        
        # 第2の残差接続
        x = x + ffn_output
        
        return x


class EnhancedWaveModel(nn.Module):
    """
    拡張版Wave Networkモデル
    
    より効率的なアーキテクチャと安定した学習のための
    様々な改良を施したモデル
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        max_seq_length: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        use_linear_attention: bool = True,
        ffn_ratio: float = 4.0
    ):
        """
        Args:
            vocab_size: 語彙サイズ
            hidden_size: 隠れ層の次元数
            num_layers: モデル層の数
            max_seq_length: 最大シーケンス長
            dropout: ドロップアウト率
            pad_token_id: パディングトークンID
            use_linear_attention: 線形計算量の注意機構を使用するか
            ffn_ratio: フィードフォワードネットワークの拡大率
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        
        # トークン埋め込み
        self.token_embedding = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )
        
        # 位置埋め込み
        self.position_embedding = nn.Embedding(
            max_seq_length, hidden_size
        )
        
        # EnhancedWave Blocks
        self.layers = nn.ModuleList([
            EnhancedWaveBlock(
                hidden_size=hidden_size,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
                use_linear_attention=use_linear_attention
            ) for _ in range(num_layers)
        ])
        
        # 最終出力用の正規化
        self.norm = RMSNorm(hidden_size)
        
        # 語彙への射影（分類器）
        self.classifier = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # 重みの共有（オプション）: 埋め込み層と分類器で重みを共有
        self.classifier.weight = self.token_embedding.weight
        
        # 重みの初期化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """重みの初期化"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                
    def create_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        位置IDを生成
        
        Args:
            input_ids: 入力トークンID [batch_size, seq_len]
            
        Returns:
            位置ID [batch_size, seq_len]
        """
        # パディングトークンを無視
        mask = input_ids.ne(self.pad_token_id).long()
        
        # 累積和で位置ID作成
        position_ids = torch.cumsum(mask, dim=1) * mask
        
        # 最大長でクランプ
        return torch.clamp(position_ids, 0, self.max_seq_length - 1)
    
    def get_input_embeddings(self) -> torch.Tensor:
        """入力埋め込み層の取得"""
        return self.token_embedding
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        モデルのフォワードパス
        
        Args:
            input_ids: 入力トークンID [batch_size, seq_len]
            attention_mask: 注意マスク（使わないが互換性のために）
            position_ids: 位置ID（オプション）
            return_dict: 辞書形式で返すか否か
            
        Returns:
            辞書またはタプルの出力
        """
        batch_size, seq_len = input_ids.shape
        
        # 位置IDの自動生成
        if position_ids is None:
            position_ids = self.create_position_ids(input_ids)
            
        # 埋め込み
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # 埋め込みの合成
        hidden_states = token_embeds + position_embeds
        
        # レイヤーを通過
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        # 最終正規化
        hidden_states = self.norm(hidden_states)
        
        # 分類器（語彙への射影）
        logits = self.classifier(hidden_states)
        
        if return_dict:
            return {
                "hidden_states": hidden_states,
                "logits": logits
            }
        else:
            return logits, hidden_states
    
    def get_classifier_weights(self) -> torch.Tensor:
        """
        分類器の重みを取得（Cut Cross Entropy用）
        """
        return self.classifier.weight
