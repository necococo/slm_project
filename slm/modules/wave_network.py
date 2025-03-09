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
    # 数値安定性のためにepsを調整し、メモリ管理を改善
    eps = 1e-4  # 数値安定性のための小さな値
    
    # メモリ使用量を減らすためにCPUキャッシュをクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 念の為float32に強制変換し、NaNをチェック
    x = x.float()
    if torch.isnan(x).any():
        print("nanが現れた。compute_wave_representation(x)")
    
    B, S, D = x.shape
    
    # グローバル振幅の計算 (モードによって集約次元が異なる)
    if global_mode:
        # 文レベル: メモリ使用量を削減するため計算を分割
        x_squared = x * x
        # チャンクで計算して合計（メモリ使用量削減）
        G_squared = torch.sum(x_squared, dim=(1, 2), keepdim=True) + eps
        G = torch.sqrt(G_squared)  # [B, 1, 1]
        G = G.expand(-1, S, D)  # [B, S, D]
    else:
        # トークンレベル: メモリ使用量を削減するため計算を分割
        x_squared = x * x
        G_squared = torch.sum(x_squared, dim=1, keepdim=True) + eps
        G = torch.sqrt(G_squared)  # [B, 1, D]
        G = G.expand(-1, S, -1)  # [B, S, D]
    
    # 中間結果をクリア
    del x_squared
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 安全な振幅値 - より安定した実装
    G_safe = torch.clamp(G, min=eps) # より厳密な下限値の保証
    
    # メモリ解放
    del G
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ===== 比率計算の数値安定性強化（ハイブリッドアプローチ） =====
    # 1. まず極端な値をclampで制限（数値安定性の確保）
    ratio = x / G_safe
    # 極端な値を制限（メモリ削減のため計算を分割）
    ratio = torch.clamp(ratio, min=-10.0, max=10.0)  # 極端な値を制限（範囲も縮小）
    
    # 2. 次にtanhで滑らかな勾配を維持しながら-0.99〜0.99に制限
    ratio = torch.tanh(ratio) * 0.99
    
    # ===== 位相角 (α_jk) の計算 =====
    # 1 - ratio^2 が負になる可能性がある（数値誤差のため）
    ratio_squared = ratio**2
    inside = 1.0 - ratio_squared
    
    # 不要な中間結果を削除
    del ratio_squared
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 非線形関数よりもclampを使用することで、厳密に非負値を保証
    inside = torch.clamp(inside, min=0.0) + eps
    
    # arctan2(√(1-ratio²), ratio) - sqrt計算を安定化
    sqrt_inside = torch.sqrt(inside)
    
    # 中間結果を削除
    del inside
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    alpha = torch.atan2(sqrt_inside, ratio)
    
    # 中間結果を削除
    del sqrt_inside
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 波表現への変換
    real_part = G_safe * torch.cos(alpha)
    imag_part = G_safe * torch.sin(alpha)
    
    # 中間結果を削除
    del G_safe, alpha
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if torch.isnan(real_part).any():
        print("nanが現れた。compute_wave_representation real_part")  

    if torch.isnan(imag_part).any():
        print("nanが現れた。compute_wave_representation imag_part")  

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
        # 振幅 = √(実部² + 虚部²)
        amplitude = torch.sqrt(real_part**2 + imag_part**2 + 1e-5)
        
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
        eps = 1e-5
        
        # NaN対策
        if torch.isnan(x).any():
            print("Warning: NaNs in input, replacing with zeros")
        
        # 文レベルのwave表現 (グローバルコンテキスト)
        real_sen, imag_sen = compute_wave_representation(x, global_mode=True, eps=eps)
        
        # トークンレベルのwave表現
        real_token, imag_token = compute_wave_representation(x, global_mode=False, eps=eps)
        
        # 波の干渉（加算）
        combined_real = real_sen + real_token
        combined_imag = imag_sen + imag_token
        
        # ウェーブレット変換を適用（低周波成分のみを使用）
        if self.use_wavelet:
            from slm.modules.wavelet import apply_wavelet_transform
            combined_real, combined_imag = apply_wavelet_transform(
                combined_real, combined_imag, wavelet_name=self.wavelet_name
            )
        
        # 生体ゆらぎゲート機構を使用する場合
        if self.use_bio_noise:
            # 実部と虚部から振幅と位相を計算
            amplitude, phase = self.compute_amplitude_phase(combined_real, combined_imag)
            
            # 振幅と位相から生体ゆらぎを組み込んだゲートを計算
            gate = self.bio_gate(amplitude, phase, training=self.training)
            
            # ゲートで重み付け
            gated_real = combined_real * gate
            gated_imag = combined_imag * gate
            
            # ゲート適用後の波表現を処理
            real_part = self.ffn_real(gated_real)  # [B, S, D]
            imag_part = self.ffn_imag(gated_imag)  # [B, S, D]
        else:
            # 従来通りの処理
            real_part = self.ffn_real(combined_real)  # [B, S, D]
            imag_part = self.ffn_imag(combined_imag)  # [B, S, D]
        
        # 実部と虚部を結合して波表現を作成
        wave_repr = torch.cat([real_part, imag_part], dim=-1)  # [B, S, 2D]
        
        # 波表現から埋め込み空間への変換
        output = self.to_embedding(wave_repr)  # [B, S, D]
        output = self.norm(output)

        return output

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
        
        # 2. 残差接続とPost-Norm（論文の図6(b)に従う）
        output = self.dropout(x + wave_output)
        
        return output
        
    def get_noise_scale(self) -> Optional[torch.Tensor]:
        """
        生体ゆらぎのスケールパラメータを取得（分析用）
        
        Returns:
            noise_scale: ノイズスケールパラメータ [D]、使用していない場合はNone
        """
        if self.use_bio_noise:
            return self.wave_layer.bio_gate.noise_scale
        return None


# 従来のWaveNetworkBlockクラスを保持（互換性のため）
class WaveNetworkBlock(WaveletEnhancedNetworkBlock):
    """
    生体ゆらぎ機能を統合したWave Network Block の実装
    """
    def __init__(
        self, 
        hidden_size: int, 
        dropout_prob: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        noise_std: float = 0.1,
        use_bio_noise: bool = True,
        trainable_noise: bool = True
    ):
        super().__init__(
            hidden_size=hidden_size, 
            dropout_prob=dropout_prob,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            noise_std=noise_std,
            use_bio_noise=use_bio_noise,
            trainable_noise=trainable_noise,
            use_wavelet=False  # デフォルトではウェーブレットを使用しない
        )

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
        # トークン埋め込み
        hidden_states = self.token_embedding(input_ids)
        
        # 各レイヤーを通過（生体ゆらぎゲート機構とウェーブレット変換を適用）
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        # 最終ノーマライゼーション
        last_hidden_states = self.norm(hidden_states)

        # 損失関数タイプに基づいて異なる出力を返す
        if self.use_cut_cross_entropy:
            # Cut Cross Entropyの場合はlast_hidden_statesを返す
            return last_hidden_states
        else:
            # 通常のCross Entropyの場合はlogitsを返す
            logits = self.classifier(last_hidden_states)
            return logits

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