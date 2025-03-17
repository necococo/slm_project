"""
Wave Network - An Ultra-Small Language Model (https://arxiv.org/abs/2411.02674v4)
この実装はFigure 6(a)(b)に基づき、波の加算による干渉表現とRoPE位置エンコーディングを使用しています。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List, Any
from slm.modules.rmsnorm import RMSNorm
from slm.modules.rope import RoPEEmbedding
from slm.modules.activations import GatedMLP  # GatedMLPをインポート
from slm.config import ModelConfig

def compute_wave_representation(x: torch.Tensor, global_mode: bool = False, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    入力テンソルを波表現（実部と虚部）に変換
    
    Args:
        x: 入力テンソル [batch_size, seq_len, dim]
        global_mode: Trueなら文レベル(グローバル)、Falseならトークンレベル(ローカル)
        eps: 数値安定性のための小さな値
        
    Returns:
        (real_part, imag_part): 波表現の実部と虚部
    """
    # 数値安定性のための定数
    eps = max(eps, 1e-6)  # より安全な最小値
    
    # 入力値のチェックと前処理
    if torch.isnan(x).any():
        # NaN値を0に置き換え
        x = torch.nan_to_num(x, nan=0.0)
    
    # 極端な値をクリップして安定化
    x = torch.clamp(x, min=-10.0, max=10.0)
    
    B, S, D = x.shape
    
    # グローバル振幅の計算 (モードによって集約次元が異なる)
    if global_mode:
        # 文レベル
        x_squared = x * x
        G_squared = torch.sum(x_squared, dim=(1, 2), keepdim=True) + eps
        G = torch.sqrt(G_squared)  # [B, 1, 1]
        G = torch.clamp(G, min=eps)  # 0に近すぎる値を避ける
        G = G.expand(-1, S, D)  # [B, S, D]
    else:
        # トークンレベル
        x_squared = x * x
        G_squared = torch.sum(x_squared, dim=1, keepdim=True) + eps
        G = torch.sqrt(G_squared)  # [B, 1, D]
        G = torch.clamp(G, min=eps)  # 0に近すぎる値を避ける
        G = G.expand(-1, S, -1)  # [B, S, D]
    
    # 比率計算
    ratio = torch.div(x, G, rounding_mode='floor')  # 安全な除算
    
    # 比率を-1と1の間に制限
    ratio = torch.clamp(ratio, min=-0.99, max=0.99)
    
    # 1 - ratio^2 の計算（数値安定性のため）
    one_minus_ratio_squared = 1.0 - ratio * ratio
    one_minus_ratio_squared = torch.clamp(one_minus_ratio_squared, min=eps)
    
    # √(1-ratio²) の計算
    sqrt_term = torch.sqrt(one_minus_ratio_squared)
    
    # arctan2(√(1-ratio²), ratio) で位相角を計算
    alpha = torch.atan2(sqrt_term, ratio)
    
    # 波表現への変換
    # 最終的な実部と虚部の計算
    real_part = G * torch.cos(alpha)
    imag_part = G * torch.sin(alpha)
    
    # 結果の検証と修正
    if torch.isnan(real_part).any() or torch.isnan(imag_part).any():
        # NaN値を0に置き換え
        real_part = torch.nan_to_num(real_part, nan=0.0)
        imag_part = torch.nan_to_num(imag_part, nan=0.0)
    
    return real_part, imag_part

class BiologicalNoiseGate(nn.Module):
    """
    振幅と位相から生体ゆらぎを組み込んだゲーティング機構
    
    生体システムでは、微小なノイズが確率共鳴などを通じて弱い信号を増幅し、
    システムの感度を高める効果が知られています。この機構はその効果を模倣します。
    """
    def __init__(self, hidden_size: int, noise_std: float = 0.1, trainable_noise: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.noise_std = noise_std
        self.trainable_noise = trainable_noise
        
        # 振幅と位相から重みを計算するための線形層
        self.W_amplitude = nn.Linear(hidden_size, hidden_size)
        self.W_phase = nn.Linear(hidden_size, hidden_size)
        
        # 学習可能なノイズスケール (trainable_noiseがTrueの場合)
        if trainable_noise:
            # ノイズ強度を制御するパラメータ（各次元ごとに異なる値を持つ）
            self.noise_scale = nn.Parameter(torch.ones(hidden_size) * noise_std)
        else:
            self.register_buffer('noise_scale', torch.ones(hidden_size) * noise_std)
        
        # バイアス項
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        振幅と位相から生体ゆらぎを組み込んだゲートを計算
        
        Args:
            amplitude: 振幅テンソル [B, S, D]
            phase: 位相テンソル [B, S, D]
            training: 訓練モードかどうか（Trueの場合のみノイズを適用）
            
        Returns:
            gate: ゲート値 [B, S, D]（値域は0〜1）
        """
        # 振幅と位相からの重み計算
        w_amp = self.W_amplitude(amplitude)
        w_phase = self.W_phase(phase)
        
        # 基本的なゲート計算（ノイズなし）
        gate_pre = w_amp + w_phase + self.bias
        
        # 訓練中かつノイズが有効な場合のみノイズを適用
        if training:
            # 生体ゆらぎを模倣した正規分布ノイズを生成
            # ノイズの大きさは各次元ごとに異なる値を持つ
            noise_shape = gate_pre.shape
            device = gate_pre.device
            noise = torch.randn(noise_shape, device=device)
            
            # 各次元ごとのノイズスケールを適用
            # self.noise_scaleをbroadcastして適用
            scaled_noise = noise * self.noise_scale.view(1, 1, -1)
            
            # ノイズを加える（生体ゆらぎの導入）
            gate_pre = gate_pre + scaled_noise
        
        # シグモイド関数でゲート値の範囲を0〜1に制限
        gate = torch.sigmoid(gate_pre)
        
        return gate

class WaveletEnhancedWaveLayer(nn.Module):
    """
    ウェーブレット変換と生体ゆらぎを組み合わせたWave Layer実装
    - 生体ゆらぎゲート機構により動的な重み付けを行う
    - Haarウェーブレット変換により低周波成分（approximation）のみを使用する
    """
    def __init__(
        self, 
        hidden_size: int, 
        dropout_prob: float = 0.1, 
        noise_std: float = 0.1, 
        use_bio_noise: bool = True, 
        trainable_noise: bool = False,  # ハイパラ探索の最適値
        use_wavelet: bool = True,
        wavelet_name: str = "haar"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_bio_noise = use_bio_noise
        self.use_wavelet = use_wavelet
        self.wavelet_name = wavelet_name
        
        # FFN拡張係数
        expansion_factor = 4
        
        # 実部・虚部それぞれのFFN
        self.ffn_real = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size * expansion_factor, hidden_size)
        )
        self.ffn_imag = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size * expansion_factor, hidden_size)
        )
        
        # 波表現から元の埋め込み空間への変換
        self.to_embedding = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = RMSNorm(hidden_size)
        
        # 生体ゆらぎゲート機構（振幅と位相から動的重みを計算）
        if use_bio_noise:
            self.bio_gate = BiologicalNoiseGate(
                hidden_size, 
                noise_std=noise_std,
                trainable_noise=trainable_noise
            )
        
    def compute_amplitude_phase(self, real_part: torch.Tensor, imag_part: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """実部と虚部から振幅と位相を計算"""
        eps = 1e-6
        # 振幅 = √(実部² + 虚部²)
        amplitude = torch.sqrt(real_part**2 + imag_part**2 + eps)
        
        # 位相 = arctan(虚部/実部)
        phase = torch.atan2(imag_part, real_part)
        
        return amplitude, phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ウェーブレットと生体ゆらぎを組み合わせたWave Layer forward pass
        
        Args:
            x: 入力テンソル [B, S, D]
            
        Returns:
            output: 処理後のテンソル [B, S, D]
        """       
        B, S, D = x.shape
        eps = 1e-6  # 安全なイプシロン値
        
        try:
            # 入力の安全性チェック
            if torch.isnan(x).any():
                x = torch.nan_to_num(x, nan=0.0)
                
            # 文レベルのwave表現 (グローバルコンテキスト)
            try:
                real_sen, imag_sen = compute_wave_representation(x, global_mode=True, eps=eps)
            except RuntimeError as e:
                # 計算エラーの場合はゼロテンソルでフォールバック
                real_sen = torch.zeros_like(x)
                imag_sen = torch.zeros_like(x)
            
            # トークンレベルのwave表現
            try:
                real_token, imag_token = compute_wave_representation(x, global_mode=False, eps=eps)
            except RuntimeError as e:
                # 計算エラーの場合はゼロテンソルでフォールバック
                real_token = torch.zeros_like(x)
                imag_token = torch.zeros_like(x)
            
            # 波の干渉（加算）- 安全に処理
            combined_real = real_sen + real_token
            combined_imag = imag_sen + imag_token
            
            # NaN値チェックと処理
            if torch.isnan(combined_real).any():
                combined_real = torch.nan_to_num(combined_real, nan=0.0)
            if torch.isnan(combined_imag).any():
                combined_imag = torch.nan_to_num(combined_imag, nan=0.0)
            
            # ウェーブレット変換を適用（低周波成分のみを使用）
            if self.use_wavelet:
                try:
                    from slm.modules.wavelet import apply_wavelet_transform
                    combined_real, combined_imag = apply_wavelet_transform(
                        combined_real, combined_imag, wavelet_name=self.wavelet_name
                    )
                except Exception:
                    # ウェーブレット変換に失敗した場合はスキップ
                    pass
            
            # 生体ゆらぎゲート機構を使用する場合
            if self.use_bio_noise:
                try:
                    # 実部と虚部から振幅と位相を計算
                    amplitude, phase = self.compute_amplitude_phase(combined_real, combined_imag)
                    
                    # 安全チェック
                    if torch.isnan(amplitude).any():
                        amplitude = torch.nan_to_num(amplitude, nan=0.0)
                    if torch.isnan(phase).any():
                        phase = torch.nan_to_num(phase, nan=0.0)
                    
                    # 振幅と位相から生体ゆらぎを組み込んだゲートを計算
                    gate = self.bio_gate(amplitude, phase, training=self.training)
                    
                    # ゲートで重み付け
                    gated_real = combined_real * gate
                    gated_imag = combined_imag * gate
                    
                    # ゲート適用後の波表現を処理
                    real_part = self.ffn_real(gated_real)  # [B, S, D]
                    imag_part = self.ffn_imag(gated_imag)  # [B, S, D]
                except Exception:
                    # エラー時は従来の処理にフォールバック
                    real_part = self.ffn_real(combined_real)  # [B, S, D]
                    imag_part = self.ffn_imag(combined_imag)  # [B, S, D]
            else:
                # 従来通りの処理
                real_part = self.ffn_real(combined_real)  # [B, S, D]
                imag_part = self.ffn_imag(combined_imag)  # [B, S, D]
            
            # NaN値チェックと処理
            if torch.isnan(real_part).any():
                real_part = torch.nan_to_num(real_part, nan=0.0)
            if torch.isnan(imag_part).any():
                imag_part = torch.nan_to_num(imag_part, nan=0.0)
            
            # 実部と虚部を結合して波表現を作成
            wave_repr = torch.cat([real_part, imag_part], dim=-1)  # [B, S, 2D]
            
            # 波表現から埋め込み空間への変換
            output = self.to_embedding(wave_repr)  # [B, S, D]
            output = self.norm(output)
            
            # 最終出力の安全性確認
            if torch.isnan(output).any():
                output = torch.nan_to_num(output, nan=0.0)
            
            return output
            
        except Exception:
            # 重大なエラーが発生した場合、入力をそのまま返して残差接続に頼る
            # 入力を安全に処理して返す
            safe_x = torch.nan_to_num(x, nan=0.0)
            return safe_x

# オリジナルの SingleWaveLayer クラスを保持（互換性のため）
class SingleWaveLayer(WaveletEnhancedWaveLayer):
    """
    生体ゆらぎを組み込んだWave Network Layer の拡張実装
    基本的な構造はFig.6(a)に基づきつつ、生体ゆらぎゲート機構を追加
    """
    def __init__(self, hidden_size: int, dropout_prob: float = 0.1, 
                 noise_std: float = 0.1, use_bio_noise: bool = True, 
                 trainable_noise: bool = True):
        super().__init__(
            hidden_size=hidden_size,
            dropout_prob=dropout_prob,
            noise_std=noise_std,
            use_bio_noise=use_bio_noise,
            trainable_noise=trainable_noise,
            use_wavelet=False  # デフォルトではウェーブレットを使用しない
        )

class WaveletEnhancedNetworkBlock(nn.Module):
    """
    ウェーブレット変換と生体ゆらぎを組み合わせたWave Network Block
    """
    def __init__(
        self, 
        hidden_size: int, 
        dropout_prob: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        noise_std: float = 0.1,
        use_bio_noise: bool = True,
        trainable_noise: bool = False,  # ハイパラ探索の最適値
        use_wavelet: bool = True,
        wavelet_name: str = "haar"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_rope = use_rope
        self.use_bio_noise = use_bio_noise
        self.use_wavelet = use_wavelet
        
        # ウェーブレット強化Wave Layer 
        self.wave_layer = WaveletEnhancedWaveLayer(
            hidden_size=hidden_size, 
            dropout_prob=dropout_prob,
            noise_std=noise_std,
            use_bio_noise=use_bio_noise,
            trainable_noise=trainable_noise,
            use_wavelet=use_wavelet,
            wavelet_name=wavelet_name
        )
        
        expansion_factor = 2
        
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(self.hidden_size * expansion_factor, self.hidden_size)
        )

        # RoPE (オプション)
        if self.use_rope:
            self.rope = RoPEEmbedding(hidden_size, max_seq_len)
                
        # 残差接続後の処理
        self.dropout = nn.Dropout(dropout_prob)
        
        # 生体ゆらぎの状態記録用（デバッグ/分析用）
        self.noise_history = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ウェーブレットと生体ゆらぎを組み込んだWave Network Block forward pass
        
        Args:
            x: 入力テンソル [B, S, D]
            
        Returns:
            処理された出力テンソル [B, S, D]
        """
        try:
            # 入力の安全性検証
            if torch.isnan(x).any():
                x = torch.nan_to_num(x, nan=0.0)
            
            # RoPE位置エンコーディングを適用（各層で適用）
            if self.use_rope:
                B, S, D = x.shape
                x_4d = x.view(B, S, 1, D)  # [B, S, 1, D] - RoPE用に形状変更
                x_4d_rope = self.rope(x_4d)  # RoPE適用 - 次元保存 [B, S, 1, D]
                wave_input = x_4d_rope.view(B, S, D)  # 元の形状に戻す [B, S, D]
            else:
                wave_input = x
            
            # 1. ウェーブレット強化Wave Layer処理
            wave_output = self.wave_layer(wave_input)  # [B, S, D]
            
            # FFN層を通す
            wave_output = self.ffn(wave_output)  # [B, S, D]
            
            # 安全性検証
            if torch.isnan(wave_output).any():
                wave_output = torch.nan_to_num(wave_output, nan=0.0)
            
            # 2. 残差接続とPost-Norm（論文の図6(b)に従う）
            output = self.dropout(x + wave_output)
            
            # 最終チェック
            if torch.isnan(output).any():
                output = torch.nan_to_num(output, nan=0.0)
                
            return output
            
        except Exception:
            # エラーが発生した場合は入力をそのまま返す（残差接続に頼る）
            return x

# WaveNetworkLMクラスに生体ゆらぎとウェーブレット変換機能を統合
class WaveNetworkLM(nn.Module):
    """
    生体ゆらぎ機能とウェーブレット変換を組み込んだWave Network モデル
    - 生体ゆらぎゲート機構により動的な重み付けを行う
    - Haarウェーブレット変換により低周波成分を使用
    - ハイパーパラメータ探索で最適化された設定を使用
    """
    def __init__(
        self,
        config: ModelConfig
    ):
        super().__init__()
        self.config = config
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        num_layers = config.num_layers
        max_seq_len = config.max_seq_len
        dropout_prob = config.dropout_prob
        use_rope = config.use_rope
        
        # 生体ゆらぎ関連の設定を取得 (ハイパラ探索の最適値をデフォルトに)
        noise_std = getattr(config, 'noise_std', 0.09377424894583282)
        use_bio_noise = getattr(config, 'use_bio_noise', True)
        trainable_noise = getattr(config, 'trainable_noise', False)
        
        # ウェーブレット関連の設定を取得
        use_wavelet = getattr(config, 'use_wavelet', True)
        wavelet_name = getattr(config, 'wavelet_name', 'haar')
        
        # cut cross entropyを使用するかどうかのフラグ
        self.use_cut_cross_entropy = getattr(config, 'use_cut_cross_entropy', True)
        
        # トークン埋め込み　nanが多発するので精緻な計算をするためfloat32にしておく
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, dtype=torch.float32)
        
        # 生体ゆらぎ+ウェーブレット機能を含むWave Network Blocksのスタック
        self.layers = nn.ModuleList([
            WaveletEnhancedNetworkBlock(
                hidden_size=hidden_size,
                dropout_prob=dropout_prob,
                use_rope=use_rope,
                max_seq_len=max_seq_len,
                noise_std=noise_std,         # 生体ゆらぎの強度（最適値）
                use_bio_noise=use_bio_noise, # 生体ゆらぎを使用するかどうか
                trainable_noise=trainable_noise,  # ノイズスケールを学習させるかどうか
                use_wavelet=use_wavelet,     # ウェーブレット変換を使用
                wavelet_name=wavelet_name    # ウェーブレットの種類
            ) for _ in range(num_layers)
        ])
        
        # 分類器（cut-cross-entropy用の重み）
        self.classifier = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # 最終ノーマライゼーション
        self.norm = RMSNorm(hidden_size)
        # 初期化
        self.apply(self._init_weights)
        
        # 設定をモデル属性として保存
        self.noise_std = noise_std
        self.use_bio_noise = use_bio_noise
        self.use_wavelet = use_wavelet
        self.wavelet_name = wavelet_name
        
    def _init_weights(self, module):
        """モデルの重みを初期化"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, BiologicalNoiseGate):
            # 生体ゆらぎゲート機構の初期化
            nn.init.normal_(module.W_amplitude.weight, mean=0.0, std=0.02)
            nn.init.normal_(module.W_phase.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
            
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        モデルのforward pass
        
        Args:
            input_ids: 入力トークンID [B, S]
            
        Returns:
            use_cut_cross_entropy=Trueの場合: hidden_states [B, S, D]
            use_cut_cross_entropy=Falseの場合: logits [B, S, V]
        """
        try:
            # トークン埋め込み
            hidden_states = self.token_embedding(input_ids)
            
            # 入力の安全性検証
            if torch.isnan(hidden_states).any():
                hidden_states = torch.nan_to_num(hidden_states, nan=0.0)
            
            # 各レイヤーを通過（生体ゆらぎゲート機構とウェーブレット変換を適用）
            for i, layer in enumerate(self.layers):
                try:
                    layer_output = layer(hidden_states)
                    if torch.isnan(layer_output).any():
                        # レイヤー出力にNaN値があれば、入力をそのまま使用
                        continue
                    hidden_states = layer_output
                except Exception:
                    # レイヤー処理でエラーが発生した場合、そのレイヤーをスキップ
                    continue
                    
            # 最終ノーマライゼーション
            last_hidden_states = self.norm(hidden_states)
            
            # NaN値チェックと処理
            if torch.isnan(last_hidden_states).any():
                last_hidden_states = torch.nan_to_num(last_hidden_states, nan=0.0)
            
            # 損失関数タイプに基づいて異なる出力を返す
            if self.use_cut_cross_entropy:
                # Cut Cross Entropyの場合はlast_hidden_statesを返す
                return last_hidden_states
            else:
                # 通常のCross Entropyの場合はlogitsを返す
                logits = self.classifier(last_hidden_states)
                
                # 最終チェック
                if torch.isnan(logits).any():
                    logits = torch.nan_to_num(logits, nan=0.0)
                    
                return logits
                
        except Exception:
            # 重大なエラー発生時は、ゼロ埋め込みを返す
            B, S = input_ids.shape
            if self.use_cut_cross_entropy:
                return torch.zeros(B, S, self.config.hidden_size, device=input_ids.device)
            else:
                return torch.zeros(B, S, self.config.vocab_size, device=input_ids.device)

    def get_classifier_weights(self) -> torch.Tensor:
        """分類器の重み (V, D)を返す"""
        return self.classifier.weight
        
    def toggle_bio_noise(self, enabled: bool = True) -> None:
        """生体ゆらぎ機能のオン/オフを切り替える"""
        self.use_bio_noise = enabled
        for layer in self.layers:
            if hasattr(layer.wave_layer, 'use_bio_noise'):
                layer.wave_layer.use_bio_noise = enabled
                
    def toggle_wavelet(self, enabled: bool = True) -> None:
        """ウェーブレット変換機能のオン/オフを切り替える"""
        self.use_wavelet = enabled
        for layer in self.layers:
            if hasattr(layer.wave_layer, 'use_wavelet'):
                layer.wave_layer.use_wavelet = enabled
                
    def set_noise_std(self, noise_std: float) -> None:
        """生体ゆらぎのノイズ強度を設定する"""
        self.noise_std = noise_std
        for layer in self.layers:
            if hasattr(layer.wave_layer, 'use_bio_noise') and layer.wave_layer.use_bio_noise:
                # 学習可能パラメータの場合は初期値だけを更新
                if hasattr(layer.wave_layer.bio_gate, 'noise_scale') and isinstance(layer.wave_layer.bio_gate.noise_scale, nn.Parameter):
                    with torch.no_grad():
                        layer.wave_layer.bio_gate.noise_scale.fill_(noise_std)
                        
    def get_learnable_noise_scales(self) -> Dict[int, torch.Tensor]:
        """各レイヤーごとの学習された生体ゆらぎスケールを取得する（分析用）"""
        result = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_noise_scale'):
                noise_scale = layer.get_noise_scale()
                if noise_scale is not None:
                    result[i] = noise_scale.detach().cpu()
        return result

    def load_state_dict(self, state_dict, strict=True):
        """状態辞書をロードする際、出力層サイズの不一致を処理する拡張メソッド"""
        try:
            # 標準のload_state_dictを試す
            return super().load_state_dict(state_dict, strict)
        except RuntimeError as e:
            # エラーメッセージが出力層サイズの不一致に関するものか確認
            if 'size mismatch' in str(e) and ('output_layer.weight' in str(e) or 'token_embedding.weight' in str(e)):
                print("出力層または埋め込み層のサイズ不一致を検出しました。サイズ調整を試みます...")
                
                # 新しい状態辞書を作成
                new_state_dict = {}
                
                # 各層を処理
                for key, value in state_dict.items():
                    if key == 'token_embedding.weight' or key == 'output_layer.weight':
                        # チェックポイントの出力層/埋め込み層のサイズ
                        checkpoint_vocab_size = value.size(0)
                        # 現在のモデルのサイズ
                        model_vocab_size = self.config.vocab_size
                        
                        print(f"チェックポイント語彙サイズ: {checkpoint_vocab_size}, モデル語彙サイズ: {model_vocab_size}")
                        
                        if checkpoint_vocab_size < model_vocab_size:
                            # チェックポイントのサイズが小さい場合は、拡張する
                            if key == 'token_embedding.weight':
                                current_shape = getattr(self, 'token_embedding').weight.shape
                                new_emb = torch.nn.init.normal_(
                                    torch.zeros(model_vocab_size, current_shape[1]),
                                    mean=0.0,
                                    std=0.02
                                ).to(value.device)
                                new_emb[:checkpoint_vocab_size] = value
                                new_state_dict[key] = new_emb
                                print(f"{key} を {checkpoint_vocab_size} から {model_vocab_size} に拡張しました")
                            elif key == 'output_layer.weight':
                                # 出力層が明示的に定義されている場合
                                if hasattr(self, 'output_layer'):
                                    current_shape = getattr(self, 'output_layer').weight.shape
                                    new_out = torch.nn.init.normal_(
                                        torch.zeros(model_vocab_size, current_shape[1]),
                                        mean=0.0,
                                        std=0.02
                                    ).to(value.device)
                                    new_out[:checkpoint_vocab_size] = value
                                    new_state_dict[key] = new_out
                                    print(f"{key} を {checkpoint_vocab_size} から {model_vocab_size} に拡張しました")
                        elif checkpoint_vocab_size > model_vocab_size:
                            # チェックポイントのサイズが大きい場合は、モデルのサイズを増やす
                            print(f"警告: チェックポイントの語彙サイズ ({checkpoint_vocab_size}) がモデル ({model_vocab_size}) より大きいです")
                            print(f"モデルの語彙サイズを {checkpoint_vocab_size} に拡張します")
                            
                            # 設定を更新
                            self.config.vocab_size = checkpoint_vocab_size
                            
                            # 埋め込み層を拡張
                            if key == 'token_embedding.weight':
                                old_emb = self.token_embedding
                                self.token_embedding = nn.Embedding(checkpoint_vocab_size, self.config.hidden_size)
                                # 既存の重みを初期化
                                with torch.no_grad():
                                    self.token_embedding.weight[:model_vocab_size].copy_(old_emb.weight)
                            
                            new_state_dict[key] = value
                        else:
                            # サイズが同じ
                            new_state_dict[key] = value
                    else:
                        # その他の層はそのまま
                        new_state_dict[key] = value
                
                # 修正した状態辞書でロード
                return super().load_state_dict(new_state_dict, strict=False)
            else:
                # その他のエラーは再発生
                raise e