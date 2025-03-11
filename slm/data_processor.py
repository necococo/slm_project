# slm/data_processor.py
# Wiki40B日本語データセット用のシンプルなデータ処理モジュール

import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, Dataset
import logging

logger = logging.getLogger(__name__)

class SimpleDataProcessor:
    """
    Wiki40B日本語データセット専用のシンプルなデータ処理クラス
    """
    def __init__(
        self, 
        tokenizer_name: str = "megagonlabs/t5-base-japanese-web", 
        max_length: int = 512,
        mask_token: str = "<mask>"
    ):
        """
        初期化
        
        Args:
            tokenizer_name: 使用するトークナイザー名
            max_length: 最大シーケンス長
            mask_token: マスクトークン
        """
        # パラメータのバリデーション
        assert isinstance(tokenizer_name, str) and tokenizer_name, "tokenizer_nameは非空の文字列である必要があります"
        assert isinstance(max_length, int) and max_length > 0, "max_lengthは正の整数である必要があります"
        assert isinstance(mask_token, str) and mask_token, "mask_tokenは非空の文字列である必要があります"
        
        # トークナイザーのロード
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # マスクトークンの設定
        if not hasattr(self.tokenizer, 'mask_token') or self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({'mask_token': mask_token})
            logger.info(f"マスクトークン '{mask_token}' を追加しました")
        
        # 設定
        self.max_length = max_length
        self.mask_token_id = self.tokenizer.mask_token_id
        self.vocab_size = self.tokenizer.vocab_size
        
        # マスクトークンIDが有効か確認
        assert self.mask_token_id is not None, "マスクトークンIDがNoneです"
        
        # 情報表示
        print(f"トークナイザー: {tokenizer_name}")
        print(f"語彙サイズ: {self.vocab_size}")
        print(f"マスクトークン: {self.tokenizer.mask_token}, ID: {self.mask_token_id}")
        
    def tokenize_text(self, text: str) -> List[int]:
        """
        テキストをトークン化
        
        Args:
            text: トークン化するテキスト
            
        Returns:
            トークンID列
        """
        # 入力のバリデーション
        assert isinstance(text, str), "textは文字列である必要があります"
        
        tokens = self.tokenizer.encode(
            text, 
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )
        
        # 結果のバリデーション
        assert isinstance(tokens, list), "エンコード結果はリストである必要があります"
        assert all(isinstance(token, int) for token in tokens), "すべてのトークンは整数である必要があります"
        
        return tokens
    
    def tokenize_batch(self, texts: List[str]) -> Dict[str, List[List[int]]]:
        """
        テキストのバッチをトークン化
        
        Args:
            texts: トークン化するテキストのリスト
            
        Returns:
            トークン化されたバッチ
        """
        tokenized = {"input_ids": [], "attention_mask": []}
        
        for text in texts:
            # トークン化
            token_ids = self.tokenize_text(text)
            
            # 注意マスクを作成（すべて1）
            attn_mask = [1] * len(token_ids)
            
            tokenized["input_ids"].append(token_ids)
            tokenized["attention_mask"].append(attn_mask)
        
        return tokenized
    
    # 標準のデコードメソッドを使用するため、独自のデコードメソッドは不要に
    
    def load_dataset(self, dataset_path: str) -> Dataset:
        """
        データセットをロード
        
        Args:
            dataset_path: データセットのパス
            
        Returns:
            ロードされたデータセット
        """
        # データセットのロード
        try:
            dataset = load_from_disk(dataset_path)
            print(f"データセットを {dataset_path} からロードしました")
            
            # データセット形式を確認
            if isinstance(dataset, dict):
                # 訓練データを取得
                if "train" in dataset:
                    train_dataset = dataset["train"]
                else:
                    print("警告: データセットに'train'スプリットがありません")
                    train_dataset = list(dataset.values())[0]
                return train_dataset
            else:
                return dataset
                
        except Exception as e:
            print(f"データセットのロードに失敗しました: {e}")
            raise
    
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """
        データセットを処理（トークン化）
        
        Args:
            dataset: 処理するデータセット
            
        Returns:
            処理されたデータセット
        """
        # データセットがtextキーを持っているか確認
        if "text" not in dataset.features:
            raise ValueError("データセットに'text'キーがありません")
        
        # トークン化関数
        def tokenize_function(examples):
            return self.tokenize_batch(examples["text"])
        
        # データセットをトークン化
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1000,
            remove_columns=["text"]
        )
        
        return tokenized_dataset
    
    def add_noise(self, tokens: torch.Tensor, noise_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        トークンにノイズを追加（一部をマスクトークンに置換）
        
        Args:
            tokens: 入力トークン [batch_size, seq_len]
            noise_ratio: ノイズの割合
            
        Returns:
            (ノイズを加えたトークン, マスク位置) のタプル
        """
        # 入力のバリデーション
        assert isinstance(tokens, torch.Tensor), "tokensはPyTorchテンソルである必要があります"
        assert tokens.dim() == 2, "tokensは2次元テンソル [batch_size, seq_len] である必要があります"
        assert 0.0 <= noise_ratio <= 1.0, "noise_ratioは0.0から1.0の間である必要があります"
        
        device = tokens.device
        batch_size, seq_len = tokens.shape
        
        # ノイズマスクを生成（各トークンをnoise_ratioの確率でマスク）
        noise_mask = torch.bernoulli(torch.full((batch_size, seq_len), noise_ratio, device=device)).bool()
        
        # 入力をコピーしてマスクトークンを追加
        noisy_tokens = tokens.clone()
        noisy_tokens[noise_mask] = self.mask_token_id
        
        # 結果のバリデーション
        assert noisy_tokens.shape == tokens.shape, "ノイズを加えたトークンの形状が元のトークンと一致しません"
        if noise_ratio > 0:
            # ノイズ率が0より大きい場合、少なくとも1つのマスクトークンが含まれるはず
            assert (noisy_tokens == self.mask_token_id).sum() > 0 or batch_size == 0, "ノイズ率が0より大きいのにマスクトークンが追加されていません"
        
        return noisy_tokens, noise_mask
    
    def prepare_diffusion_batch(self, batch: Dict[str, torch.Tensor], noise_ratio: float = 0.15) -> Dict[str, torch.Tensor]:
        """
        Diffusionモデル用のバッチを準備
        
        Args:
            batch: 入力バッチ
            noise_ratio: ノイズの割合
            
        Returns:
            Diffusionモデル用のバッチ
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"] if "attention_mask" in batch else torch.ones_like(input_ids)
        
        # ノイズを追加
        noisy_tokens, noise_mask = self.add_noise(input_ids, noise_ratio)
        
        # マスクされた部分のみを予測対象にする（他は-100で無視）
        labels = torch.ones_like(input_ids) * -100
        labels[noise_mask] = input_ids[noise_mask]
        
        return {
            "input_ids": noisy_tokens,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def test_decode_with_mask(self, text: str, mask_ratio: float = 0.2) -> None:
        """
        テキストのマスキングとデコードのテスト
        
        Args:
            text: テスト用テキスト
            mask_ratio: マスク割合
        """
        # テキストのトークン化
        tokens = self.tokenize_text(text)
        token_tensor = torch.tensor([tokens])
        
        # ノイズ追加
        noisy_tokens, _ = self.add_noise(token_tensor, mask_ratio)
        noisy_tokens_list = noisy_tokens[0].tolist()
        
        # デコード
        original_text = self.tokenizer.decode(tokens)
        # 標準のdecode関数を使用
        standard_decoded = self.tokenizer.decode(noisy_tokens_list)
        
        # 結果表示
        print("オリジナルテキスト:")
        print(original_text)
        print("\n標準デコード結果:")
        print(standard_decoded)
        
        # マスクトークンの数を確認
        mask_count = noisy_tokens_list.count(self.mask_token_id)
        print(f"\nマスクトークン数: {mask_count}/{len(noisy_tokens_list)} ({mask_count/len(noisy_tokens_list)*100:.1f}%)")