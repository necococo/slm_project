"""
データセットの準備と前処理を行うユーティリティ
様々なデータソースからモデル学習用のデータセットを構築
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from datasets import load_dataset, Dataset as HFDataset
from transformers import PreTrainedTokenizer, AutoTokenizer

logger = logging.getLogger(__name__)

def prepare_mlm_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 512,
    text_column_name: str = "text",
    cache_dir: Optional[str] = None
) -> Tuple[Dataset, Dataset]:
    """
    マスク言語モデリング用のデータセットを準備
    
    Args:
        dataset_name: データセット名 (wikitext, bookcorpus, oscar, など)
        dataset_config: データセット設定名やバージョン
        tokenizer: 使用するトークナイザー
        max_seq_length: 最大シーケンス長
        text_column_name: テキスト列の名前
        cache_dir: キャッシュディレクトリ
        
    Returns:
        (train_dataset, val_dataset): 訓練用と検証用のデータセットのタプル
    """
    logger.info(f"データセット {dataset_name}/{dataset_config} をロード中...")
    
    # データソースに応じて読み込み方法を分岐
    if dataset_name == "wikitext":
        raw_datasets = load_dataset(dataset_name, dataset_config, cache_dir=cache_dir)
        
        # train/validationがない場合は分割する
        if "validation" not in raw_datasets:
            raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1)
            train_dataset = raw_datasets["train"]
            val_dataset = raw_datasets["test"]
        else:
            train_dataset = raw_datasets["train"]
            val_dataset = raw_datasets["validation"]
            
    elif dataset_name == "bookcorpus":
        raw_datasets = load_dataset(dataset_name, cache_dir=cache_dir)
        # train/validationに分割
        split_datasets = raw_datasets["train"].train_test_split(test_size=0.1)
        train_dataset = split_datasets["train"]
        val_dataset = split_datasets["test"]
        
    elif dataset_name == "oscar":
        raw_datasets = load_dataset(dataset_name, dataset_config, cache_dir=cache_dir)
        # train/validationに分割
        split_datasets = raw_datasets["train"].train_test_split(test_size=0.1)
        train_dataset = split_datasets["train"]
        val_dataset = split_datasets["test"]
        
    elif dataset_name == "custom":
        # カスタムデータセット（CSVやJSONなど）
        raw_datasets = load_dataset("csv" if dataset_config.endswith(".csv") else "json", 
                                   data_files=dataset_config, cache_dir=cache_dir)
        split_datasets = raw_datasets["train"].train_test_split(test_size=0.1)
        train_dataset = split_datasets["train"]
        val_dataset = split_datasets["test"]
    
    else:
        raise ValueError(f"サポートされていないデータセット: {dataset_name}")

    # テキスト列の名前を確認
    column_names = train_dataset.column_names
    if text_column_name not in column_names:
        text_column_name = column_names[0]  # 最初の列を使用
        logger.warning(f"指定されたテキスト列がありません。代わりに {text_column_name} を使用します")

    # トークン化関数
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], 
                        truncation=True, 
                        max_length=max_seq_length,
                        padding=False,
                        return_special_tokens_mask=True)

    # トークン化の実行
    logger.info("データセットをトークン化中...")
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        desc="訓練データのトークン化",
    )
    
    val_tokenized = val_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        desc="検証データのトークン化",
    )

    logger.info(f"訓練データ: {len(train_tokenized)}サンプル, 検証データ: {len(val_tokenized)}サンプル")
    
    return train_tokenized, val_tokenized

class TextDataset(Dataset):
    """
    テキストファイルベースのデータセット
    """
    def __init__(
        self, 
        file_path: str, 
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
        line_by_line: bool = False
    ):
        """
        Args:
            file_path: テキストファイルのパス
            tokenizer: 使用するトークナイザー
            max_seq_length: 最大シーケンス長
            line_by_line: 1行ごとに1サンプルとするか
        """
        assert os.path.isfile(file_path), f"ファイルが見つかりません: {file_path}"
        
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        logger.info(f"テキストファイルを読み込み中: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            if line_by_line:
                # 行ごとに分割
                lines = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
                self.examples = []
                for line in lines:
                    encodings = tokenizer(line, truncation=True, max_length=max_seq_length)
                    self.examples.append(encodings)
            else:
                # 大きなテキスト全体をトークン化
                text = f.read()
                tokens = tokenizer.tokenize(text)
                
                # 適切な長さのチャンクに分割
                self.examples = []
                i = 0
                while i < len(tokens) - max_seq_length:
                    chunk = tokens[i:i + max_seq_length]
                    encodings = tokenizer.encode_plus(
                        chunk, 
                        truncation=False, 
                        add_special_tokens=True
                    )
                    self.examples.append(encodings)
                    i += max_seq_length // 2  # 半分オーバーラップ
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}
