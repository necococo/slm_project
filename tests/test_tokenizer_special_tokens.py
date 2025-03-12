#!/usr/bin/env python
# coding: utf-8
# tests/test_tokenizer_special_tokens.py
# トークナイザーの特殊トークンテスト

import os
import sys
import argparse

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slm.tokenizer import load_tokenizer

def test_tokenizer_functions(tokenizer):
    """トークナイザーの基本機能をテスト"""
    print("\n=== トークナイザー基本機能テスト ===")
    
    test_text = "これはトークナイザーのテストです。特殊トークンが正しく機能するか確認します。"
    print(f"テスト文: {test_text}")
    
    # 基本的なエンコード・デコード
    token_ids = tokenizer.encode(test_text, add_special_tokens=False)
    decoded_text = tokenizer.decode(token_ids)
    
    print(f"トークンID: {token_ids}")
    
    if hasattr(tokenizer, 'convert_ids_to_tokens'):
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        print(f"トークン一覧: {tokens}")
    
    print(f"デコード結果: {decoded_text}")
    print(f"エンコード→デコード一致: {'はい' if test_text == decoded_text else 'いいえ'}")
    
    return token_ids

def test_special_tokens(tokenizer):
    """特殊トークン機能をテスト"""
    print("\n=== 特殊トークン機能テスト ===")
    
    # 利用可能な特殊トークンを確認
    special_tokens = {
        "<mask>": getattr(tokenizer, "mask_token", None),
        "<pad>": getattr(tokenizer, "pad_token", None),
        "<bos>": getattr(tokenizer, "bos_token", None),
        "<eos>": getattr(tokenizer, "eos_token", None),
        "<unk>": getattr(tokenizer, "unk_token", None),
        "<cls>": getattr(tokenizer, "cls_token", None),
        "<sep>": getattr(tokenizer, "sep_token", None),
    }
    
    # 特殊トークン情報の表示
    print("特殊トークン一覧:")
    for name, token in special_tokens.items():
        if token:
            token_id = None
            if name == "<mask>" and hasattr(tokenizer, "mask_token_id"):
                token_id = tokenizer.mask_token_id
            elif name == "<pad>" and hasattr(tokenizer, "pad_token_id"):
                token_id = tokenizer.pad_token_id
            elif name == "<bos>" and hasattr(tokenizer, "bos_token_id"):
                token_id = tokenizer.bos_token_id
            elif name == "<eos>" and hasattr(tokenizer, "eos_token_id"):
                token_id = tokenizer.eos_token_id
            elif name == "<unk>" and hasattr(tokenizer, "unk_token_id"):
                token_id = tokenizer.unk_token_id
            elif name == "<cls>" and hasattr(tokenizer, "cls_token_id"):
                token_id = tokenizer.cls_token_id
            elif name == "<sep>" and hasattr(tokenizer, "sep_token_id"):
                token_id = tokenizer.sep_token_id
                
            print(f"  {name}: '{token}' (ID: {token_id})")
    
    # 語彙サイズの確認
    if hasattr(tokenizer, "vocab_size"):
        print(f"\n語彙サイズ: {tokenizer.vocab_size}")
    
    # 全特殊トークンリストの確認
    if hasattr(tokenizer, "all_special_tokens"):
        print(f"全特殊トークン: {tokenizer.all_special_tokens}")
        print(f"全特殊トークンID: {tokenizer.all_special_ids}")
    
    # <mask>トークンのテスト
    if getattr(tokenizer, "mask_token", None):
        test_mask_token(tokenizer)
    
    # <pad>トークンのテスト
    if getattr(tokenizer, "pad_token", None):
        test_pad_token(tokenizer)
    
    # <unk>トークンのテスト
    if getattr(tokenizer, "unk_token", None):
        test_unk_token(tokenizer)

def test_mask_token(tokenizer):
    """<mask>トークンの詳細テスト"""
    print("\n--- <mask>トークンテスト ---")
    
    # マスクトークンのエンコード
    mask_token = tokenizer.mask_token
    mask_token_id = tokenizer.mask_token_id
    print(f"<mask>トークン: '{mask_token}' (ID: {mask_token_id})")
    
    # マスクトークンを含むテキストをエンコード
    test_text = f"これは{mask_token}のテストです。"
    token_ids = tokenizer.encode(test_text, add_special_tokens=False)
    
    # トークン情報表示
    if hasattr(tokenizer, 'convert_ids_to_tokens'):
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        print(f"マスクを含むテキスト: {test_text}")
        print(f"トークンID: {token_ids}")
        print(f"トークン一覧: {tokens}")
    
    # マスクトークンの位置を確認
    mask_positions = [i for i, id in enumerate(token_ids) if id == mask_token_id]
    print(f"マスクトークン位置: {mask_positions}")
    
    # デコード結果確認
    decoded_text = tokenizer.decode(token_ids)
    print(f"デコード結果: {decoded_text}")
    print(f"マスクトークンが保持されている: {'はい' if mask_token in decoded_text else 'いいえ'}")
    
    # 標準的な文をエンコードし、特定の位置をマスクに置き換え
    normal_text = "これはマスク置換のテストです。"
    normal_ids = tokenizer.encode(normal_text, add_special_tokens=False)
    
    # "マスク"の位置を特定 (実際のインデックスはトークナイザーによって異なる可能性あり)
    if hasattr(tokenizer, 'convert_ids_to_tokens'):
        tokens = tokenizer.convert_ids_to_tokens(normal_ids)
        print(f"\n元のテキスト: {normal_text}")
        print(f"元のトークン: {tokens}")
    
    # "マスク"文字列が含まれる部分を探す
    mask_target_idx = 2  # 近似位置として仮定
    
    # マスク置換
    masked_ids = normal_ids.copy()
    masked_ids[mask_target_idx] = mask_token_id
    
    # 置換結果確認
    if hasattr(tokenizer, 'convert_ids_to_tokens'):
        masked_tokens = tokenizer.convert_ids_to_tokens(masked_ids)
        print(f"マスク置換後のトークン: {masked_tokens}")
    
    masked_text = tokenizer.decode(masked_ids)
    print(f"マスク置換後のテキスト: {masked_text}")

def test_pad_token(tokenizer):
    """<pad>トークンの詳細テスト"""
    print("\n--- <pad>トークンテスト ---")
    
    # パッドトークンのエンコード
    pad_token = tokenizer.pad_token
    pad_token_id = tokenizer.pad_token_id
    print(f"<pad>トークン: '{pad_token}' (ID: {pad_token_id})")
    
    # テキストをエンコードしてパディング
    test_text = "これはパディングのテストです。"
    token_ids = tokenizer.encode(test_text, add_special_tokens=False)
    
    # パディングを追加
    padded_ids = token_ids + [pad_token_id] * 3
    
    # トークン情報表示
    if hasattr(tokenizer, 'convert_ids_to_tokens'):
        tokens = tokenizer.convert_ids_to_tokens(padded_ids)
        print(f"パディング前テキスト: {test_text}")
        print(f"パディング後トークンID: {padded_ids}")
        print(f"パディング後トークン一覧: {tokens}")
    
    # デコード結果確認
    decoded_text = tokenizer.decode(padded_ids)
    print(f"デコード結果: {decoded_text}")
    
    # アテンションマスクを使用した場合の挙動（実際のモデルでは使用される）
    attention_mask = [1] * len(token_ids) + [0] * 3
    print(f"アテンションマスク: {attention_mask}")
    print("注: アテンションマスクが0の位置のパッドトークンは、モデルの計算で無視されます")

def test_unk_token(tokenizer):
    """<unk>トークンの詳細テスト"""
    print("\n--- <unk>トークンテスト ---")
    
    # UNKトークンのエンコード
    unk_token = tokenizer.unk_token
    unk_token_id = tokenizer.unk_token_id
    print(f"<unk>トークン: '{unk_token}' (ID: {unk_token_id})")
    
    # 語彙外の文字/単語をテスト
    oov_chars = "😊🚀👍"  # 絵文字などの特殊文字
    token_ids = tokenizer.encode(oov_chars, add_special_tokens=False)
    
    # トークン情報表示
    if hasattr(tokenizer, 'convert_ids_to_tokens'):
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        print(f"語彙外文字: {oov_chars}")
        print(f"トークンID: {token_ids}")
        print(f"トークン一覧: {tokens}")
    
    # UNKトークンの出現回数確認
    unk_count = token_ids.count(unk_token_id) if unk_token_id is not None else 0
    print(f"UNKトークン出現回数: {unk_count}")
    
    # デコード結果確認
    decoded_text = tokenizer.decode(token_ids)
    print(f"デコード結果: {decoded_text}")
    print(f"元の文字列が保持されている: {'はい' if oov_chars == decoded_text else 'いいえ'}")

def parse_args():
    parser = argparse.ArgumentParser(description="トークナイザーの特殊トークンテスト")
    
    parser.add_argument("--tokenizer_name", type=str, default="megagonlabs/t5-base-japanese-web",
                       help="使用するトークナイザー名")
    
    return parser.parse_args()

def main():
    # コマンドライン引数の解析
    args = parse_args()
    
    # トークナイザーのロード
    print(f"トークナイザー '{args.tokenizer_name}' をロード中...")
    tokenizer = load_tokenizer(args.tokenizer_name)
    print(f"トークナイザーをロードしました: {tokenizer.__class__.__name__}")
    
    # 基本的な機能テスト
    test_tokenizer_functions(tokenizer)
    
    # 特殊トークンテスト
    test_special_tokens(tokenizer)
    
    print("\nトークナイザーのテストが完了しました")

if __name__ == "__main__":
    main()