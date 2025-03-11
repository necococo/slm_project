# slm/tokenizer.py
# SLM用の簡素化されたトークナイザーユーティリティ - megagonlabs/t5-base-japanese-web専用

from transformers import AutoTokenizer
from typing import Dict, Any, List

def load_tokenizer(model_name: str = "megagonlabs/t5-base-japanese-web") -> AutoTokenizer:
    """
    megagonlabs/t5-base-japanese-webトークナイザーをロードする簡易関数
    
    Args:
        model_name: 使用するトークナイザー名（デフォルトはmegagonlabs/t5-base-japanese-web）
        
    Returns:
        設定済みのトークナイザー
    """
    # トークナイザーをロード
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # マスクトークンがない場合は追加
    if not hasattr(tokenizer, 'mask_token') or tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '<mask>'})
        print(f"マスクトークン '<mask>' を追加しました。マスクトークンID: {tokenizer.mask_token_id}")
    
    # BOSトークンがない場合は追加 
    if not hasattr(tokenizer, 'bos_token') or tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '<s>'})
        print(f"BOSトークン '<s>' を追加しました。")
    
    # トークナイザー情報を表示
    print(f"トークナイザー情報:")
    print(f"  名前: {model_name}")
    print(f"  語彙サイズ: {tokenizer.vocab_size}")
    print(f"  マスクトークン: {tokenizer.mask_token}, ID: {tokenizer.mask_token_id}")
    
    return tokenizer

def tokenize_batch(tokenizer: AutoTokenizer, texts: List[str], max_length: int = 512) -> Dict[str, List[List[int]]]:
    """
    テキストのバッチをトークン化する簡易関数
    
    Args:
        tokenizer: トークナイザー
        texts: トークン化するテキストのリスト
        max_length: 最大シーケンス長
        
    Returns:
        トークン化されたバッチ（input_ids, attention_maskを含む辞書）
    """
    tokenized = {"input_ids": [], "attention_mask": []}
    
    for text in texts:
        # トークン化
        token_ids = tokenizer.encode(
            text, 
            add_special_tokens=False,
            max_length=max_length,
            truncation=True
        )
        
        # 注意マスクを作成（すべて1）
        attn_mask = [1] * len(token_ids)
        
        tokenized["input_ids"].append(token_ids)
        tokenized["attention_mask"].append(attn_mask)
    
    return tokenized