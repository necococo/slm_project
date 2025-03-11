#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import sentencepiece as spm

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slm.wiki40b_ja_dataset import load_tokenizer

def test_advanced_tokenization(tokenizer, text):
    """
    SentencePieceトークナイザーの詳細な挙動確認
    """
    print("\n=== SentencePieceトークナイザー詳細テスト ===")
    print(f"テスト文: {text}")
    
    # 1. 基本的なエンコード・デコード
    print("\n1. 基本的なエンコード・デコード:")
    ids = tokenizer.encode(text, out_type=int)
    str_tokens = tokenizer.encode(text, out_type=str)
    decoded = tokenizer.decode(ids)
    
    print(f"トークンID: {ids}")
    print(f"文字列トークン: {str_tokens}")
    print(f"デコード結果: {decoded}")
    print(f"一致?: {text == decoded}")
    
    # 2. 先頭トークンを除去して試す
    if len(ids) > 1 and str_tokens[0] == '▁':
        print("\n2. 先頭の特殊トークン(▁)を除去した場合:")
        decoded_without_first = tokenizer.decode(ids[1:])
        print(f"トークンID: {ids[1:]}")
        print(f"文字列トークン: {str_tokens[1:]}")
        print(f"デコード結果: {decoded_without_first}")
    
    # 3. ピース単位で分解して表示
    print("\n3. ピース単位の詳細:")
    for i, (token_id, token_str) in enumerate(zip(ids, str_tokens)):
        piece_info = f"トークン {i}: ID={token_id}, 文字列=「{token_str}」"
        # ▁ がある場合は特別表示
        if '▁' in token_str:
            piece_info += f" (▁を含む: 単語先頭を示す特殊文字)"
        print(piece_info)
    
    # 4. 文字単位でエンコード・デコード
    print("\n4. 文字単位の詳細:")
    for i, char in enumerate(text):
        char_ids = tokenizer.encode(char, out_type=int)
        char_tokens = tokenizer.encode(char, out_type=str)
        char_decoded = tokenizer.decode(char_ids)
        print(f"文字 {i} '{char}': ID={char_ids}, トークン={char_tokens}, デコード='{char_decoded}'")
        
    # 5. NBest候補取得（SentencePieceの高度な機能）
    if hasattr(tokenizer, 'nbest_encode'):
        print("\n5. NBest分割候補:")
        try:
            nbest_size = 5
            nbest_results = tokenizer.nbest_encode(text, nbest_size=nbest_size)
            for i, candidate in enumerate(nbest_results):
                print(f"候補 {i+1}: {candidate}")
        except Exception as e:
            print(f"NBest候補取得失敗: {e}")
    
    # 6. コード側の問題でなくSentencePieceの仕様である可能性を示す
    print("\n6. SentencePieceの一般的な挙動について:")
    print("- SentencePieceは単語の区切りを表現するために'▁'(U+2581)を使います")
    print("- 日本語などスペースのない言語では、デコード時に▁が特殊処理される場合があります")
    print("- 完全な往復変換(roundtrip)は保証されていない場合があります")
    print("- トークナイズ方法によっては、特に文頭や特殊文字周辺で不一致が発生することがあります")

if __name__ == "__main__":
    # コマンドライン引数からデータディレクトリとモデルプレフィックスを受け取る
    import argparse
    parser = argparse.ArgumentParser(description='SentencePieceトークナイザーの詳細テスト')
    parser.add_argument('--data_dir', type=str, default="/content/drive/MyDrive/slm/data/wiki40b_ja", 
                        help='モデルファイルのディレクトリ')
    parser.add_argument('--model_prefix', type=str, default="sp_jwiki", 
                        help='モデルファイルの接頭辞')
    parser.add_argument('--text', type=str, default="これはトークナイザの機能を確認するためのテスト文章です。", 
                        help='テストするテキスト')
    args = parser.parse_args()
    
    # トークナイザーをロード
    try:
        tokenizer = load_tokenizer(args.data_dir, args.model_prefix)
        print(f"トークナイザーをロードしました: {args.data_dir}/{args.model_prefix}.model")
        print(f"語彙サイズ: {tokenizer.get_piece_size()}")
        
        # テスト実行
        test_advanced_tokenization(tokenizer, args.text)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)