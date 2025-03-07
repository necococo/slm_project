#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
トークナイザーと埋め込み表現の検証スクリプト
- トークナイザーの動作確認 (encode → decode の一貫性)
- 単語埋め込みの関係性分析 (例: 王 - 男 + 女 = 女王)
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 親ディレクトリをパスに追加してslmモジュールをインポートできるようにする
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slm.config import ModelConfig, PathsConfig
from slm.modules.wave_network import WaveNetworkLM
from slm.utils import load_checkpoint

def check_tokenizer(tokenizer):
    """トークナイザーの動作確認を実行"""
    print("=== トークナイザー動作確認 ===")
    print(f"トークナイザークラス: {tokenizer.__class__.__name__}")
    print(f"語彙サイズ: {tokenizer.vocab_size}")
    
    # サンプルテキスト
    sample_texts = [
        "これはテスト文です。",
        "自然言語処理は面白いです。",
        "人工知能と機械学習について学んでいます。",
        "王様と女王様が王宮でパーティーを開きました。"
    ]
    
    for text in sample_texts:
        print("\n----")
        print(f"原文: {text}")
        
        # トークン化
        tokens = tokenizer.tokenize(text) if hasattr(tokenizer, 'tokenize') else None
        if tokens:
            print(f"トークン: {tokens}")
        
        # エンコード
        ids = tokenizer.encode(text, add_special_tokens=False) \
            if hasattr(tokenizer, 'encode') else tokenizer(text).input_ids
        print(f"ID: {ids}")
        
        # デコード
        decoded = tokenizer.decode(ids) if hasattr(tokenizer, 'decode') else tokenizer.decode(ids)
        print(f"デコード結果: {decoded}")
        
        # エンコード→デコードの一貫性チェック
        if text == decoded:
            print("✓ エンコード→デコードの一貫性: 完全一致")
        else:
            print(f"× エンコード→デコードの一貫性: 不一致\n  原文: {text}\n  復元: {decoded}")

def analyze_embeddings(model, tokenizer):
    """単語埋め込みの分析"""
    print("\n=== 単語埋め込み分析 ===")
    
    # 単語ペア関係性テスト用語彙
    word_pairs = [
        # 基本的な性別関係
        ("王", "女王"),
        ("男", "女"),
        ("俳優", "女優"),
        ("父", "母"),
        ("息子", "娘"),
        ("おじさん", "おばさん"),
        ("兄", "姉"),
        ("彼", "彼女"),
        # 敬称関係
        ("さん", "様"),
        # 動物関係
        ("犬", "猫"),
        ("ライオン", "ライオネス")
    ]
    
    # 単語をIDに変換
    word_to_id = {}
    for pair in word_pairs:
        for word in pair:
            if hasattr(tokenizer, 'encode'):
                word_to_id[word] = tokenizer.encode(word, add_special_tokens=False)[0]
            else:
                word_to_id[word] = tokenizer(word, add_special_tokens=False).input_ids[0]
    
    # 埋め込み層を取得
    embedding_layer = model.token_embedding.weight.detach().cpu().numpy()
    
    # ベクトル取得＆コサイン類似度計算
    vectors = {}
    for word, word_id in word_to_id.items():
        vectors[word] = embedding_layer[word_id]
    
    # ベクトル間の関係性検証
    print("\n単語ベクトル関係性:")
    for word1, word2 in word_pairs:
        if word1 in vectors and word2 in vectors:
            # コサイン類似度
            similarity = cosine_similarity(vectors[word1], vectors[word2])
            print(f"「{word1}」と「{word2}」の類似度: {similarity:.4f}")
    
    # 埋め込みベクトルの可視化
    visualize_embeddings(vectors, "単語埋め込み空間の2D投影")
    
    # 単語アナロジー（類推）テスト
    print("\n単語アナロジーテスト:")
    analogy_tests = [
        ("王", "男", "女", "女王"),  # 王 - 男 + 女 = 女王
        ("父", "男", "女", "母"),     # 父 - 男 + 女 = 母
    ]
    
    for word1, word2, word3, expected in analogy_tests:
        if all(w in vectors for w in [word1, word2, word3, expected]):
            # word1 - word2 + word3 ≈ expected
            result_vector = vectors[word1] - vectors[word2] + vectors[word3]
            
            # 最も近い単語を検索
            best_similarity = -float("inf")
            best_word = None
            for word, vec in vectors.items():
                sim = cosine_similarity(result_vector, vec)
                if sim > best_similarity:
                    best_similarity = sim
                    best_word = word
            
            # 正解との類似度
            expected_sim = cosine_similarity(result_vector, vectors[expected])
            
            print(f"「{word1} - {word2} + {word3}」= 「{best_word}」 (期待: 「{expected}」)")
            print(f"  正解との類似度: {expected_sim:.4f}")

def cosine_similarity(vec1, vec2):
    """コサイン類似度を計算"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def visualize_embeddings(vectors, title):
    """埋め込みベクトルを2次元に圧縮して可視化"""
    words = list(vectors.keys())
    embedding_matrix = np.array([vectors[word] for word in words])
    
    # t-SNEで次元削減
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(words)-1))
    reduced_vectors = tsne.fit_transform(embedding_matrix)
    
    # 可視化
    plt.figure(figsize=(12, 10))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7)
    
    # 単語ラベル
    for i, word in enumerate(words):
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), 
                     fontsize=12, alpha=0.8)
    
    plt.title(title)
    plt.savefig("embeddings_visualization.png")
    print(f"埋め込み可視化を embeddings_visualization.png に保存しました")

def main():
    """メイン関数"""
    # 設定
    paths_config = PathsConfig()
    model_config = ModelConfig()
    
    # トークナイザー読み込み（AutoTokenizerを直接使用）
    print("トークナイザーを読み込み中...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(paths_config.tokenizer_name)
        print(f"トークナイザーを読み込みました: {paths_config.tokenizer_name}")
        print(f"語彙サイズ: {len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else tokenizer.vocab_size}")
    except Exception as e:
        print(f"トークナイザーの読み込みに失敗しました: {e}")
        return
    
    # トークナイザーの動作確認（モデルがなくても実行可能）
    check_tokenizer(tokenizer)
    
    # 埋め込み分析はモデルが必要なため、チェックポイントが存在する場合のみ実行
    checkpoint_path = os.path.join(paths_config.checkpoint_dir, "final_model.pt")
    if os.path.exists(checkpoint_path):
        # モデルの構成にトークナイザーをセット
        model_config.set_tokenizer(tokenizer)
        
        # モデル初期化
        print("\nモデルを初期化中...")
        model = WaveNetworkLM(model_config)
        
        # チェックポイントから読み込み
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"モデルをロードしました: {checkpoint_path}")
            
            # モデルを評価モードに設定
            model.eval()
            
            # 埋め込み分析
            analyze_embeddings(model, tokenizer)
        except Exception as e:
            print(f"モデルのロードに失敗しました: {e}")
            print("埋め込み分析はスキップします。")
    else:
        print(f"\nチェックポイントファイルがありません: {checkpoint_path}")
        print("埋め込み分析をスキップし、トークナイザーの検証のみ実行しました。")
        print("埋め込み分析を実行するには、学習済みモデルが必要です。")

if __name__ == "__main__":
    main()
