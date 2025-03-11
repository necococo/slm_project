# slm/simple_diffusion.py
# シンプルなDiffusionモデル実装

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union

class SimpleMaskingDiffusion(nn.Module):
    """
    マスキングベースのシンプルなテキスト拡散モデル
    """
    def __init__(
        self, 
        mask_token_id: int, 
        mask_prob_min: float = 0.0,
        mask_prob_max: float = 0.9,
        timesteps: int = 10,
    ):
        """
        初期化
        
        Args:
            mask_token_id: マスクトークンのID
            mask_prob_min: 最小マスク確率
            mask_prob_max: 最大マスク確率
            timesteps: タイムステップ数
        """
        # パラメータのバリデーション
        assert isinstance(mask_token_id, int) and mask_token_id >= 0, "mask_token_idは非負の整数である必要があります"
        assert isinstance(mask_prob_min, float) and 0.0 <= mask_prob_min < 1.0, "mask_prob_minは0.0以上1.0未満の浮動小数点数である必要があります"
        assert isinstance(mask_prob_max, float) and 0.0 < mask_prob_max <= 1.0, "mask_prob_maxは0.0より大きく1.0以下の浮動小数点数である必要があります"
        assert mask_prob_min <= mask_prob_max, "mask_prob_minはmask_prob_max以下である必要があります"
        assert isinstance(timesteps, int) and timesteps > 0, "timestepsは正の整数である必要があります"
        
        super().__init__()
        self.mask_token_id = mask_token_id
        self.mask_prob_min = mask_prob_min
        self.mask_prob_max = mask_prob_max
        self.timesteps = timesteps
        
        # マスク確率のスケジュール（線形増加）
        mask_probs = torch.linspace(mask_prob_min, mask_prob_max, timesteps)
        self.register_buffer("mask_probs", mask_probs)
        
        # マスク確率のバリデーション
        assert len(mask_probs) == timesteps, "マスク確率の数がタイムステップ数と一致しません"
        assert torch.all(mask_probs >= 0) and torch.all(mask_probs <= 1), "マスク確率は0から1の範囲内である必要があります"
        
        print(f"SimpleMaskingDiffusion初期化: マスクトークンID={mask_token_id}, タイムステップ数={timesteps}")
        print(f"マスク確率: {mask_prob_min:.2f}～{mask_prob_max:.2f}")
    
    def add_noise(
        self, 
        tokens: torch.Tensor, 
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        指定されたタイムステップでトークンにノイズを追加
        
        Args:
            tokens: 入力トークン [batch_size, seq_len]
            t: タイムステップ [batch_size] (0 〜 timesteps-1)
            
        Returns:
            (ノイズを加えたトークン, マスク位置) のタプル
        """
        # 入力のバリデーション
        assert isinstance(tokens, torch.Tensor), "tokensはPyTorchテンソルである必要があります"
        assert tokens.dim() == 2, "tokensは2次元テンソル [batch_size, seq_len] である必要があります"
        assert isinstance(t, torch.Tensor), "tはPyTorchテンソルである必要があります"
        assert t.dim() == 1, "tは1次元テンソル [batch_size] である必要があります"
        assert tokens.size(0) == t.size(0), "tokensとtのバッチサイズが一致しません"
        
        # タイムステップの値が正しい範囲内かチェック
        assert torch.all(t >= 0) and torch.all(t < self.timesteps), f"tは0以上{self.timesteps-1}以下である必要があります"
        
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # バッチ内の各サンプルに対して異なるマスク確率を適用
        mask_probs_t = torch.stack([self.mask_probs[min(t_i.item(), self.timesteps-1)] for t_i in t])
        
        # マスク位置をランダムに選択
        mask_prob_matrix = mask_probs_t.unsqueeze(-1).expand(batch_size, seq_len)
        mask = torch.bernoulli(mask_prob_matrix).bool()
        
        # マスクを適用
        noisy_tokens = tokens.clone()
        noisy_tokens[mask] = self.mask_token_id
        
        # 結果のバリデーション
        assert noisy_tokens.shape == tokens.shape, "ノイズを加えたトークンの形状が元のトークンと一致しません"
        # マスク位置には必ずマスクトークンIDが入っているか確認
        assert torch.all(noisy_tokens[mask] == self.mask_token_id), "マスク位置にマスクトークンIDが入っていません"
        
        return noisy_tokens, mask
    
    def forward(
        self, 
        tokens: torch.Tensor, 
        t: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        フォワードパス（ノイズを追加してラベルを生成）
        
        Args:
            tokens: 入力トークン [batch_size, seq_len]
            t: タイムステップ [batch_size] (指定がなければランダムに生成)
            
        Returns:
            ノイズを加えたトークンとラベルを含む辞書
        """
        # 入力のバリデーション
        assert isinstance(tokens, torch.Tensor), "tokensはPyTorchテンソルである必要があります"
        assert tokens.dim() == 2, "tokensは2次元テンソル [batch_size, seq_len] である必要があります"
        if t is not None:
            assert isinstance(t, torch.Tensor), "tはPyTorchテンソルまたはNoneである必要があります"
            assert t.dim() == 1, "tは1次元テンソル [batch_size] である必要があります"
            assert tokens.size(0) == t.size(0), "tokensとtのバッチサイズが一致しません"
        
        batch_size = tokens.shape[0]
        device = tokens.device
        
        # タイムステップが指定されていない場合はランダムに選択
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        # ノイズを追加
        noisy_tokens, mask = self.add_noise(tokens, t)
        
        # マスクされた位置のみを予測対象にする（他は-100で無視）
        labels = torch.ones_like(tokens) * -100
        labels[mask] = tokens[mask]
        
        # 結果のバリデーション
        assert noisy_tokens.shape == tokens.shape, "ノイズを加えたトークンの形状が元のトークンと一致しません"
        assert labels.shape == tokens.shape, "ラベルの形状が元のトークンと一致しません"
        # マスク位置と予測対象が一致するか確認
        assert torch.all((mask == (labels != -100))), "マスク位置と予測対象が一致しません"
        # マスクトークンIDの位置と予測対象が一致するか確認
        assert torch.all(((noisy_tokens == self.mask_token_id) == (labels != -100))), "マスクトークンIDの位置と予測対象が一致しません"
        # ラベルの値が元のトークンと一致するか確認
        assert torch.all(labels[mask] == tokens[mask]), "マスク位置のラベル値が元のトークンと一致しません"
        
        return {
            "input_ids": noisy_tokens,
            "labels": labels,
            "timesteps": t,
            "mask": mask
        }
    
    def sample(
        self, 
        model: nn.Module, 
        tokens: torch.Tensor, 
        timesteps: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_intermediate: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        ノイズ除去によるサンプリング
        
        Args:
            model: 言語モデル
            tokens: 初期トークン（ノイズあり）[batch_size, seq_len]
            timesteps: サンプリングに使用するタイムステップのリスト
            temperature: サンプリング温度
            top_k: Top-kサンプリング用のk
            top_p: nucleus samplingのp値
            return_intermediate: 中間ステップも返すかどうか
            
        Returns:
            サンプリングされたトークン（または中間ステップを含むリスト）
        """
        # 入力のバリデーション
        assert isinstance(model, nn.Module), "modelはnn.Moduleである必要があります"
        assert isinstance(tokens, torch.Tensor), "tokensはPyTorchテンソルである必要があります"
        assert tokens.dim() == 2, "tokensは2次元テンソル [batch_size, seq_len] である必要があります"
        
        if timesteps is not None:
            assert isinstance(timesteps, list), "timestepsはリストである必要があります"
            assert all(isinstance(t, int) for t in timesteps), "timestepsの要素はすべて整数である必要があります"
            assert all(0 <= t < self.timesteps for t in timesteps), f"timestepsの要素はすべて0以上{self.timesteps-1}以下である必要があります"
        
        assert isinstance(temperature, (int, float)), "temperatureは数値である必要があります"
        if top_k is not None:
            assert isinstance(top_k, int) and top_k > 0, "top_kは正の整数である必要があります"
        if top_p is not None:
            assert isinstance(top_p, float) and 0.0 < top_p <= 1.0, "top_pは0.0より大きく1.0以下の浮動小数点数である必要があります"
        
        device = tokens.device
        if timesteps is None:
            # デフォルトでは最大ノイズから徐々に除去
            timesteps = list(range(self.timesteps - 1, -1, -1))
        
        # 各ステップでのトークンを保存
        intermediate = []
        current_tokens = tokens.clone()
        
        # 少なくとも1つのマスクトークンが含まれていることを確認
        initial_mask_count = (current_tokens == self.mask_token_id).sum().item()
        if initial_mask_count == 0:
            print(f"警告: 入力にマスクトークン(ID: {self.mask_token_id})が含まれていません。サンプリングは行われません。")
        
        for step, t in enumerate(timesteps):
            # 現在のタイムステップでのバッチを作成
            t_tensor = torch.full((tokens.shape[0],), t, device=device)
            
            # マスク位置を特定（ノイズが加えられた位置）
            masked_positions = (current_tokens == self.mask_token_id)
            
            # マスク位置がない場合は終了
            if not masked_positions.any():
                break
            
            # モデルで予測
            with torch.no_grad():
                logits = model(current_tokens)  # [batch_size, seq_len, vocab_size]
                
                # ロジットのバリデーション
                assert isinstance(logits, torch.Tensor), "モデルの出力はPyTorchテンソルである必要があります"
                expected_shape = (current_tokens.size(0), current_tokens.size(1), -1)
                assert logits.dim() == 3, f"モデル出力は3次元テンソル [batch_size, seq_len, vocab_size] である必要があります"
                assert logits.size(0) == expected_shape[0] and logits.size(1) == expected_shape[1], \
                    f"モデル出力の形状 {logits.shape} が期待される形状 {expected_shape} と一致しません"
            
            # マスク位置のみで予測
            masked_logits = logits[masked_positions]  # [num_masks, vocab_size]
            
            # サンプリング温度を適用
            if temperature > 0:
                masked_logits = masked_logits / temperature
            
            # top-kサンプリング
            if top_k is not None and top_k > 0:
                top_k = min(top_k, masked_logits.size(-1))
                indices_to_remove = masked_logits < torch.topk(masked_logits, top_k)[0][..., -1, None]
                masked_logits[indices_to_remove] = float('-inf')
            
            # top-pサンプリング
            if top_p is not None and top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(masked_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 累積確率がtop_pを超えるトークンを除外
                sorted_indices_to_remove = cumulative_probs > top_p
                # 最初のトークンは必ず含める
                sorted_indices_to_remove[..., 0] = False
                
                # インデックスを元の順序に戻す
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, 
                    index=sorted_indices, 
                    src=sorted_indices_to_remove
                )
                masked_logits[indices_to_remove] = float('-inf')
            
            # softmaxを適用して確率分布を得る
            probs = F.softmax(masked_logits, dim=-1)
            
            # 確率分布のバリデーション
            assert torch.all(probs >= 0), "確率分布に負の値が含まれています"
            assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1))), "確率分布の合計が1でない"
            
            # 次のトークンをサンプリング
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # サンプリング結果のバリデーション
            assert next_tokens.dim() == 1, "サンプリング結果は1次元テンソルである必要があります"
            assert next_tokens.size(0) == masked_positions.sum(), "サンプリング結果の数がマスク位置の数と一致しません"
            
            # マスク位置に予測トークンを代入
            current_tokens_flat = current_tokens.view(-1)
            masked_positions_flat = masked_positions.view(-1)
            current_tokens_flat[masked_positions_flat] = next_tokens
            
            # マスクトークンが正しく置き換えられたか確認
            assert torch.all(current_tokens[masked_positions] != self.mask_token_id), "マスク位置のトークンが置き換えられていません"
            
            # 中間結果を保存
            if return_intermediate:
                intermediate.append(current_tokens.clone())
            
        # 結果を返す
        if return_intermediate:
            return intermediate
        else:
            return current_tokens


# テスト関数
def test_simple_masking_diffusion():
    """SimpleMaskingDiffusionクラスの動作をテスト"""
    print("SimpleMaskingDiffusionテスト開始")
    
    # パラメータ
    mask_token_id = 4
    batch_size = 2
    seq_len = 10
    timesteps = 5
    
    # モックトークン
    tokens = torch.randint(10, 100, (batch_size, seq_len))
    
    # Diffusionモデル
    diffusion = SimpleMaskingDiffusion(
        mask_token_id=mask_token_id,
        mask_prob_min=0.1,
        mask_prob_max=0.5,
        timesteps=timesteps
    )
    
    # 各タイムステップでのマスク率をテスト
    for t in range(timesteps):
        t_tensor = torch.full((batch_size,), t)
        noisy_tokens, mask = diffusion.add_noise(tokens, t_tensor)
        
        # マスク率を計算
        mask_ratio = mask.float().mean().item()
        expected_ratio = diffusion.mask_probs[t].item()
        
        print(f"タイムステップ {t}: マスク率 {mask_ratio:.2f} (期待値 {expected_ratio:.2f})")
        
        # マスクがちゃんと適用されているか確認
        assert torch.all(noisy_tokens[mask] == mask_token_id)
    
    # forwardメソッドのテスト
    for t in range(timesteps):
        t_tensor = torch.full((batch_size,), t)
        result = diffusion(tokens, t_tensor)
        
        # 各要素を確認
        assert "input_ids" in result
        assert "labels" in result
        assert "timesteps" in result
        assert "mask" in result
        
        # マスク部分だけがラベル化されているか確認
        mask = result["mask"]
        labels = result["labels"]
        
        assert torch.all(labels[mask] >= 0)
        assert torch.all(labels[~mask] == -100)
        
        # ラベル値が正しいか確認
        assert torch.all(labels[mask] == tokens[mask])
    
    print("SimpleMaskingDiffusionテスト完了")

if __name__ == "__main__":
    test_simple_masking_diffusion()