#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
How:
    評価用のスクリプト。Googleドライブからデータを高速アクセス可能な
    ローカルディレクトリにコピーし、トークナイザーをロードします。

Why not:
    Googleドライブは直接アクセスすると遅いため、高速なローカルストレージに
    データをコピーして処理速度を向上させます。
"""

import os
import shutil
from pathlib import Path
import time
from typing import Optional, Union, Dict, Any
from transformers import AutoTokenizer, PreTrainedTokenizer


def copy_data_to_fast_storage(
    source_path: str = "/content/drive/MyDrive/slm/data/fujiki/wiki40b_ja",
    target_path: str = "/content/fast_data/",
) -> str:
    """
    Googleドライブからローカルの高速ストレージにデータをコピーします。

    Args:
        source_path: コピー元のパス
        target_path: コピー先のパス
    
    Returns:
        str: コピー先の完全なパス
    """    
    # ターゲットディレクトリが存在しない場合は作成
    target_dir = Path(target_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # コピー先のフルパス
    full_target_path = target_dir / Path(source_path).name
    
    # コピー元が存在するか確認
    if not Path(source_path).exists():
        raise FileNotFoundError(f"コピー元のパスが見つかりません: {source_path}")
    
    # すでにコピー先にデータがある場合はスキップするオプションを提供
    if full_target_path.exists():
        print(f"コピー先にデータがすでに存在します: {full_target_path}")
        return str(full_target_path)
    else:
        shutil.copytree(source_path, full_target_path)
        print(f"データをコピーしました: {source_path} -> {full_target_path}")
        return str(full_target_path)


def load_tokenizer(
    tokenizer_path: str,
    use_fast: bool = True,
) -> Optional[PreTrainedTokenizer]:
    """
    トークナイザーをロードします。
    
    Args:
        tokenizer_path: トークナイザーが保存されているパス
        use_fast: 高速バージョンを使用するかどうか
        
    Returns:
        Optional[PreTrainedTokenizer]: ロードされたトークナイザーのインスタンス
    
    Why not:
        直接パスを指定することで、様々なディレクトリ構造に対応できるようにしています。
        エラー処理を組み込むことで、トークナイザーが見つからない場合でも安全に処理を継続します。
    """
    if "AutoTokenizer" not in globals():
        print("transformersライブラリがインポートされていません。")
        return None
    
    tokenizer_path = Path(tokenizer_path)
    
    if not tokenizer_path.exists():
        print(f"トークナイザーのパスが見つかりません: {tokenizer_path}")
        return None
    
    try:
        print(f"トークナイザーをロードしています: {tokenizer_path}")
        
        # AutoTokenizerを使ってトークナイザーをロード
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            use_fast=use_fast
        )
        
        print(f"トークナイザーが正常にロードされました")
        print(f"語彙サイズ: {len(tokenizer)}")
        
        # マスクトークンが存在しない場合は追加
        # Why not: MLMタスクなどではマスクトークンが必須なため、
        # 明示的に追加して後続処理での問題を防止します
        if not hasattr(tokenizer, 'mask_token') or tokenizer.mask_token is None:
            tokenizer.add_special_tokens({'mask_token': '<mask>'})
            print(f"マスクトークン '<mask>' を追加しました。マスクトークンID: {tokenizer.mask_token_id}")
        else:
            print(f"マスクトークン: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
        
        # 特殊トークンの情報を表示
        if hasattr(tokenizer, 'special_tokens_map'):
            print(f"特殊トークン: {tokenizer.special_tokens_map}")
        else:
            # フォールバック：一般的な特殊トークンを個別に取得
            special_tokens = {}
            for token_name, token_attr in [
                ("PAD", "pad_token"),
                ("UNK", "unk_token"),
                ("BOS", "bos_token"),
                ("EOS", "eos_token"),
                ("SEP", "sep_token"),
                ("CLS", "cls_token"),
                ("MASK", "mask_token")
            ]:
                if hasattr(tokenizer, token_attr):
                    token_value = getattr(tokenizer, token_attr, None)
                    if token_value is not None:
                        special_tokens[token_name] = token_value
            
            print(f"特殊トークン: {special_tokens}")
        
        return tokenizer
    
    except Exception as e:
        print(f"トークナイザーのロード中にエラーが発生しました: {str(e)}")
        return None


def prepare_environment() -> Dict[str, Any]:
    """
    評価環境を準備します。データのコピーとトークナイザーのロードを行います。
    
    Returns:
        Dict[str, Any]: 環境設定の結果情報（パスとトークナイザー）
    """
    result = {}
    
    # データをコピー
    data_path = copy_data_to_fast_storage()
    result["data_path"] = data_path
    
    # トークナイザーをロード
    tokenizer_path = os.path.join(data_path, "tokenizers")
    tokenizer = load_tokenizer(tokenizer_path)
    result["tokenizer"] = tokenizer
    
    return result


if __name__ == "__main__":
    try:
        # 評価環境を準備
        env = prepare_environment()
        print("\n評価環境の準備が完了しました。")
        print(f"データパス: {env.get('data_path', 'N/A')}")
        print(f"トークナイザー: {'ロード成功' if env.get('tokenizer') is not None else 'ロード失敗'}")
    except Exception as e:
        print(f"スクリプト実行中にエラーが発生しました: {str(e)}")
