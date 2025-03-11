#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
megagonlabs/t5-base-japanese-webのSentencePieceトークナイザーを試すスクリプト
"""

import sys
import os
from transformers import AutoTokenizer

# 親ディレクトリをパスに追加してslmモジュールをインポートできるようにする
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slm.tokenizer import JapaneseTokenizer

def test_transformers_tokenizer(text):
    """transformers のトークナイザーを直接テスト"""
    print("\n=== transformers から直接ロードしたトークナイザー ===")
    tokenizer = AutoTokenizer.from_pretrained("megagonlabs/t5-base-japanese-web")
    
    # トークン化
    tokens = tokenizer.tokenize(text)
    print(f"トークン: {tokens}")
    
    # エンコード
    encoded = tokenizer.encode(text, add_special_tokens=False)
    print(f"トークンID: {encoded}")
    
    # デコード
    decoded = tokenizer.decode(encoded)
    print(f"デコード結果: {decoded}")
    
    # 一致確認
    print(f"元のテキストと一致?: {text == decoded}")
    if text != decoded:
        print(f"  元のテキスト: {text}")
        print(f"  デコード結果: {decoded}")

def test_japanese_tokenizer_wrapper(text):
    """JapaneseTokenizerラッパーを使ったテスト"""
    print("\n=== JapaneseTokenizerラッパーを使用 ===")
    
    # まずHuggingFaceからトークナイザーを取得
    hf_tokenizer = AutoTokenizer.from_pretrained("megagonlabs/t5-base-japanese-web")
    
    # JapaneseTokenizerラッパーに変換
    jp_tokenizer = JapaneseTokenizer.from_pretrained_tokenizer(hf_tokenizer)
    
    # エンコード
    token_ids = jp_tokenizer.encode(text)
    print(f"トークンID: {token_ids}")
    
    # デコード
    decoded = jp_tokenizer.decode(token_ids)
    print(f"デコード結果: {decoded}")
    
    # 一致確認
    print(f"元のテキストと一致?: {text == decoded}")
    if text != decoded:
        print(f"  元のテキスト: {text}")
        print(f"  デコード結果: {decoded}")

def demonstrate_tokenizer_use_in_code():
    """JapaneseTokenizerをコードで使用する例を示す"""
    print("\n=== JapaneseTokenizerの使用例 ===")
    
    # JapaneseTokenizerをHuggingFaceモデルから初期化
    tokenizer = JapaneseTokenizer(hf_model="megagonlabs/t5-base-japanese-web")
    
    # サンプルテキスト
    texts = [
        "これはトークナイザーのテストです。日本語Wikipediaで学習されたモデルを使います。",
        "自然言語処理は面白いですね。トークン化はその基本的な処理の一つです。"
    ]
    
    for text in texts:
        print(f"\nテキスト: {text}")
        
        # エンコード
        token_ids = tokenizer.encode(text)
        print(f"トークンID: {token_ids}")
        
        # デコード
        decoded = tokenizer.decode(token_ids)
        print(f"デコード結果: {decoded}")
        
        # 一致確認
        print(f"元のテキストと一致?: {text == decoded}")

if __name__ == "__main__":
    # テストするテキスト
    test_text = "これはトークナイザーのテストです。日本語Wikipediaで学習されたモデルを使います。"
    
    # transformersトークナイザーを直接テスト
    test_transformers_tokenizer(test_text)
    
    # JapaneseTokenizerラッパーを使ったテスト
    test_japanese_tokenizer_wrapper(test_text)
    
    # コード内での使用例
    demonstrate_tokenizer_use_in_code()