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
    else:
        try:
            print(f"トークナイザーをロードしています: {tokenizer_path}")
            
            # AutoTokenizerを使ってトークナイザーをロード
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                use_fast=use_fast
            )
            
            print(f"トークナイザーが正常にロードされました")
            print(f"語彙サイズ: {len(tokenizer)}")
            
            # マスクトークンの存在確認
            has_mask_token = hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None
            
            if not has_mask_token:
                print("マスクトークンが存在しません。")
                
                # 特定の範囲のIDに対応するトークンを確認（デバッグ用）
                print("\n特定範囲のトークンID確認:")
                for i in range(31990, 32020):
                    if i < len(tokenizer):
                        decoded_text = tokenizer.decode([i], skip_special_tokens=False)
                        print(f"ID: {i} -> '{decoded_text}'")
                
                # マスクトークンを追加
                print("\nマスクトークンを追加します...")
                # トークナイザーのタイプによって異なる処理
                if hasattr(tokenizer, "add_special_tokens"):
                    tokenizer.add_special_tokens({'mask_token': '<mask>'})
                    print(f"マスクトークン '<mask>' を追加しました。マスクトークンID: {tokenizer.mask_token_id}")
                else:
                    print("このトークナイザーはadd_special_tokensメソッドをサポートしていません。")
            else:
                print(f"マスクトークンが存在します: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
            
            return tokenizer
    
        except Exception as e:
            print(f"トークナイザーのロード中にエラーが発生しました: {str(e)}")
            return None


def evaluate_mask_token(tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    """
    マスクトークンの機能を評価します。
    
    Args:
        tokenizer: 評価対象のトークナイザー
        
    Returns:
        Dict[str, Any]: 評価結果
    
    How:
        - マスクトークンの有無を確認
        - マスクトークンのエンコード・デコード動作をテスト
        - 様々なマスクパターンを試行
    """
    results = {
        "has_mask_token": False,
        "mask_token": None,
        "mask_token_id": None,
        "encoding_tests": [],
        "collator_compatible": False
    }
    
    # マスクトークンの確認
    has_mask_token = hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None
    results["has_mask_token"] = has_mask_token
    
    if has_mask_token:
        results["mask_token"] = tokenizer.mask_token
        results["mask_token_id"] = tokenizer.mask_token_id
        print(f"マスクトークン: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
    else:
        print("マスクトークンが存在しません")
        return results
    
    # エンコード・デコードテスト
    test_cases = [
        f"{tokenizer.mask_token}かもしれない",
        f"これは{tokenizer.mask_token}テストです",
        f"複数の{tokenizer.mask_token}を{tokenizer.mask_token}テスト"
    ]
    
    print("\nマスクトークンのエンコード・デコードテスト:")
    for test in test_cases:
        encoded = tokenizer.encode(test, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        
        # マスクトークンIDが含まれているか確認
        contains_mask = tokenizer.mask_token_id in encoded
        
        result = {
            "text": test,
            "encoded": encoded,
            "decoded": decoded,
            "contains_mask_token": contains_mask
        }
        results["encoding_tests"].append(result)
        
        print(f"テスト: '{test}'")
        print(f"エンコード: {encoded}")
        print(f"デコード: '{decoded}'")
        print(f"マスクトークン含有: {'✓' if contains_mask else '✗'}")
        print("-" * 40)
    
    # CustomCollatorとの互換性テスト
    try:
        from types import SimpleNamespace
        from slm.collator import CustomCollator
        
        # モデル設定のモック
        mock_config = SimpleNamespace(max_seq_len=128)
        
        # サンプルデータ
        test_text = f"これは{tokenizer.mask_token}テストです。"
        encoded = tokenizer.encode(test_text)
        test_sample = {
            "input_ids": encoded,
            "attention_mask": [1] * len(encoded)
        }
        
        # Collatorの初期化
        collator = CustomCollator(
            tokenizer=tokenizer,
            model_config=mock_config,
            mlm=True,
            mlm_probability=0.15
        )
        
        # バッチ処理
        batch = collator([test_sample])
        
        # マスクトークンがあるか確認
        mask_count = (batch["input_ids"] == tokenizer.mask_token_id).sum().item()
        
        results["collator_compatible"] = True
        results["mask_count"] = mask_count
        
        print("\nCollatorテスト:")
        print(f"入力テキスト: '{test_text}'")
        print(f"バッチ形状: {batch['input_ids'].shape}")
        print(f"マスクされたトークン数: {mask_count}")
        
    except ImportError:
        print("\nslm.collatorモジュールをインポートできませんでした。")
        results["collator_error"] = "Module not found"
    except Exception as e:
        print(f"\nCollatorテストでエラーが発生しました: {str(e)}")
        results["collator_error"] = str(e)
    
    return results


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
    
    # マスクトークンの評価
    if tokenizer is not None:
        print("\n=== マスクトークンの評価 ===")
        mask_evaluation = evaluate_mask_token(tokenizer)
        result["mask_evaluation"] = mask_evaluation
    
    return result


if __name__ == "__main__":
    try:
        # 評価環境を準備
        env = prepare_environment()
        print("\n評価環境の準備が完了しました。")
        print(f"データパス: {env.get('data_path', 'N/A')}")
        print(f"トークナイザー: {'ロード成功' if env.get('tokenizer') is not None else 'ロード失敗'}")
        
        # マスク評価結果のサマリー
        if "mask_evaluation" in env:
            eval_result = env["mask_evaluation"]
            print("\n=== マスクトークン評価サマリー ===")
            print(f"マスクトークン存在: {'あり' if eval_result.get('has_mask_token') else 'なし'}")
            if eval_result.get('has_mask_token'):
                print(f"マスクトークン: {eval_result.get('mask_token')} (ID: {eval_result.get('mask_token_id')})")
            print(f"Collator互換性: {'✓' if eval_result.get('collator_compatible') else '✗'}")
            if 'collator_error' in eval_result:
                print(f"Collatorエラー: {eval_result['collator_error']}")
    except Exception as e:
        print(f"スクリプト実行中にエラーが発生しました: {str(e)}")
