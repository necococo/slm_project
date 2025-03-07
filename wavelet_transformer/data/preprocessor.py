"""
テキストデータの前処理ユーティリティ
Wavelet Transformerに最適化された前処理関数を提供
"""
import os
import re
import unicodedata
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import torch
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    テキストデータの前処理を行うクラス
    
    テキストの正規化、クリーニング、フォーマット変換などを行います
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        lowercase: bool = True,
        remove_accents: bool = True,
        clean_text: bool = True
    ):
        """
        Args:
            tokenizer: 使用するトークナイザー
            max_length: 最大シーケンス長
            lowercase: 小文字化するか
            remove_accents: アクセント記号を除去するか
            clean_text: テキストクリーニングを行うか
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lowercase = lowercase
        self.remove_accents = remove_accents
        self.clean_text = clean_text
    
    def normalize_text(self, text: str) -> str:
        """
        テキストの正規化（小文字化、アクセント除去など）
        
        Args:
            text: 入力テキスト
            
        Returns:
            正規化されたテキスト
        """
        if self.lowercase:
            text = text.lower()
            
        if self.remove_accents:
            # Unicode正規化でアクセント記号を分離してから除去
            text = unicodedata.normalize('NFKD', text)
            text = ''.join([c for c in text if not unicodedata.combining(c)])
            
        return text
    
    def clean(self, text: str) -> str:
        """
        テキストクリーニング（不要な文字の除去など）
        
        Args:
            text: 入力テキスト
            
        Returns:
            クリーニング後のテキスト
        """
        if not self.clean_text:
            return text
            
        # 余分な空白の削除
        text = re.sub(r'\s+', ' ', text)
        
        # 制御文字の削除
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        # HTML/XMLタグの削除
        text = re.sub(r'<[^>]+>', '', text)
        
        # 連続する句読点の削除
        text = re.sub(r'([.,!?;:])\1+', r'\1', text)
        
        return text.strip()
    
    def prepare_for_wavelet(self, text: str) -> str:
        """
        Wavelet Transformer向けに特殊な前処理を行う
        
        Args:
            text: 入力テキスト
            
        Returns:
            Wavelet Transformer向けに最適化されたテキスト
        """
        # 正規化とクリーニング
        text = self.normalize_text(text)
        text = self.clean(text)
        
        # Wavelet特有の処理: 段落の区切りを明示
        text = re.sub(r'\n\s*\n', ' [PAR] ', text)
        
        # 文の区切りが明確になるよう調整
        text = re.sub(r'([.!?])\s', r'\1 [SENT] ', text)
        
        return text
    
    def process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        テキストを処理してモデル入力形式に変換
        
        Args:
            text: 入力テキスト
            
        Returns:
            モデル入力用の辞書 (input_ids, attention_mask)
        """
        # Wavelet向けの前処理
        processed_text = self.prepare_for_wavelet(text)
        
        # トークン化
        encoded = self.tokenizer(
            processed_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoded
    
    def batch_process(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        複数のテキストを一括処理
        
        Args:
            texts: 入力テキストのリスト
            
        Returns:
            バッチ処理された入力辞書
        """
        processed_texts = [self.prepare_for_wavelet(text) for text in texts]
        
        # バッチトークン化
        encoded = self.tokenizer(
            processed_texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoded
    
    def create_wavelet_features(
        self,
        input_ids: torch.Tensor,
        num_wavelet_levels: int = 3
    ) -> Dict[str, torch.Tensor]:
        """
        Wavelet特徴量を生成
        
        Args:
            input_ids: 入力ID [batch_size, seq_len]
            num_wavelet_levels: ウェーブレット分解レベル
            
        Returns:
            ウェーブレット特徴を含む辞書
        """
        # 1次元信号に変換するための埋め込み取得
        embeddings = self.tokenizer.get_input_embeddings()(input_ids)
        batch_size, seq_len, hidden_size = embeddings.shape
        
        # バッチ処理用のダミー特徴
        # 実際の実装では、PyWaveletsを使ったウェーブレット分解などが必要
        wavelet_features = {
            'input_ids': input_ids,
            'embeddings': embeddings,
            'wavelet_level_markers': torch.ones(batch_size, num_wavelet_levels, dtype=torch.float32)
        }
        
        return wavelet_features


class WaveletAugmenter:
    """
    Wavelet特性を活かしたデータ拡張クラス
    """
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Args:
            tokenizer: 使用するトークナイザー
        """
        self.tokenizer = tokenizer
        
    def frequency_masking(
        self,
        input_ids: torch.Tensor,
        mask_prob: float = 0.1,
        mask_freq_range: Tuple[float, float] = (0.2, 0.8)
    ) -> torch.Tensor:
        """
        特定の周波数帯域をマスクする拡張
        
        Args:
            input_ids: 入力ID [batch_size, seq_len]
            mask_prob: マスクする確率
            mask_freq_range: マスクする周波数範囲 (低周波, 高周波)
            
        Returns:
            拡張された入力ID
        """
        batch_size, seq_len = input_ids.shape
        masked_input_ids = input_ids.clone()
        
        # バッチごとに処理
        for i in range(batch_size):
            # 拡張の確率
            if np.random.random() > mask_prob:
                continue
                
            # ランダムな周波数範囲を選択
            low_freq = mask_freq_range[0]
            high_freq = mask_freq_range[1]
            
            # フーリエ変換のシミュレーション：特定の位置をマスク
            mask_start = int(seq_len * low_freq)
            mask_end = int(seq_len * high_freq)
            mask_length = mask_end - mask_start
            
            # マスク位置を選択
            mask_pos = list(range(mask_start, mask_end))
            np.random.shuffle(mask_pos)
            mask_pos = mask_pos[:int(mask_length * mask_prob)]
            
            # 選択された位置をマスク
            for pos in mask_pos:
                if pos < seq_len:
                    masked_input_ids[i, pos] = self.tokenizer.mask_token_id
                    
        return masked_input_ids
    
    def scale_augmentation(
        self,
        input_ids: torch.Tensor,
        prob: float = 0.1,
        scales: List[int] = [2, 4]
    ) -> torch.Tensor:
        """
        マルチスケール拡張：異なるスケールでのサンプリング
        
        Args:
            input_ids: 入力ID [batch_size, seq_len]
            prob: 拡張を適用する確率
            scales: 適用するスケール係数
            
        Returns:
            拡張された入力ID
        """
        if np.random.random() > prob:
            return input_ids
            
        batch_size, seq_len = input_ids.shape
        augmented_ids = input_ids.clone()
        
        # ランダムなスケールを選択
        scale = np.random.choice(scales)
        
        # スケーリング適用（ダウンサンプリング後にアップサンプリング）
        for i in range(batch_size):
            # ダウンサンプリング（単純化のため、間隔を空けてサンプリング）
            downsampled = input_ids[i, ::scale]
            
            # アップサンプリング（同じ値を繰り返し）
            upsampled = []
            for idx in range(len(downsampled)):
                upsampled.extend([downsampled[idx]] * scale)
                
            # 元の長さに合わせる
            upsampled = upsampled[:seq_len]
            if len(upsampled) < seq_len:
                upsampled.extend([self.tokenizer.pad_token_id] * (seq_len - len(upsampled)))
                
            # 更新
            augmented_ids[i] = torch.tensor(upsampled, device=input_ids.device)
            
        return augmented_ids


def prepare_wavelet_inputs(
    texts: Union[str, List[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    augmentation: bool = False,
    return_tensors: str = "pt"
) -> Dict[str, torch.Tensor]:
    """
    Wavelet Transformer用の入力を準備するユーティリティ関数
    
    Args:
        texts: 入力テキストまたはテキストのリスト
        tokenizer: 使用するトークナイザー
        max_length: 最大シーケンス長
        augmentation: データ拡張を適用するか
        return_tensors: 返却形式
        
    Returns:
        モデル入力用の辞書
    """
    # 単一テキストをリストに変換
    if isinstance(texts, str):
        texts = [texts]
        
    # 前処理を適用
    preprocessor = TextPreprocessor(
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # バッチ処理
    inputs = preprocessor.batch_process(texts)
    
    # データ拡張（必要な場合）
    if augmentation:
        augmenter = WaveletAugmenter(tokenizer)
        # 確率的に拡張を適用
        if np.random.random() < 0.3:  # 30%の確率で適用
            inputs['input_ids'] = augmenter.frequency_masking(inputs['input_ids'])
    
    return inputs
