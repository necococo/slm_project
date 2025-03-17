#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
How:
    評価用のスクリプト。Googleドライブからデータを高速アクセス可能な
    ローカルディレクトリにコピーし、トークナイザーとモデルをロードします。

Why not:
    Googleドライブは直接アクセスすると遅いため、高速なローカルストレージに
    データをコピーして処理速度を向上させます。
"""

import os
import shutil
from pathlib import Path
import time
from typing import Optional, Dict, Any, Tuple
import torch
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
    # ソースパスがファイルの場合とディレクトリの場合で処理を分ける
    # Why not: ファイルとディレクトリで異なるコピー処理が必要なため
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"コピー元のパスが見つかりません: {source_path}")
    
    # ターゲットディレクトリが存在しない場合は作成
    target_dir = Path(target_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # ソースがファイルかディレクトリかによって処理を分ける
    if source.is_file():
        # ファイルの場合
        target_file = target_dir / source.name
        if target_file.exists():
            print(f"コピー先にファイルがすでに存在します: {target_file}")
        else:
            shutil.copy2(source, target_file)
            print(f"ファイルをコピーしました: {source} -> {target_file}")
        return str(target_file)
    else:
        # ディレクトリの場合
        full_target_path = target_dir / source.name
        if full_target_path.exists():
            print(f"コピー先にディレクトリがすでに存在します: {full_target_path}")
        else:
            shutil.copytree(source, full_target_path)
            print(f"ディレクトリをコピーしました: {source} -> {full_target_path}")
        return str(full_target_path)


def load_tokenizer(
    tokenizer_path: str,
    use_fast: bool = True,
    add_mask_token: bool = True
) -> Optional[PreTrainedTokenizer]:
    """
    トークナイザーをロードします。
    
    Args:
        tokenizer_path: トークナイザーが保存されているパス
        use_fast: 高速バージョンを使用するかどうか
        add_mask_token: マスクトークンがない場合に追加するかどうか
        
    Returns:
        Optional[PreTrainedTokenizer]: ロードされたトークナイザーのインスタンス
    
    Why not:
        トークナイザーにマスクトークンがない場合、明示的に追加することで
        後続のMLM処理などでエラーが発生しないようにします。
    """
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
        has_mask_token = hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None
        
        if not has_mask_token and add_mask_token:
            if hasattr(tokenizer, "add_special_tokens"):
                tokenizer.add_special_tokens({'mask_token': '<mask>'})
                print(f"マスクトークン '<mask>' を追加しました。マスクトークンID: {tokenizer.mask_token_id}")
            else:
                print("警告: このトークナイザーはマスクトークンをサポートしていません。")
        elif has_mask_token:
            print(f"マスクトークン: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
        
        return tokenizer
    
    except Exception as e:
        print(f"トークナイザーのロード中にエラーが発生しました: {str(e)}")
        return None


def load_model(
    model_path: str,
    device: Optional[torch.device] = None
) -> Optional[Any]:
    """
    学習済みモデルを読み込みます。
    
    Args:
        model_path: モデル重みファイルのパス
        device: モデルをロードするデバイス
        
    Returns:
        Optional[Any]: ロードされたモデル
    
    How:
        モデル重みをロードし、シンプルなデフォルト設定で
        WaveNetworkモデルを初期化します。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルパスの確認
    model_file = Path(model_path)
    if not model_file.exists() or not model_file.is_file():
        print(f"モデルファイルが見つかりません: {model_path}")
        return None
    
    try:
        print(f"モデル重みをロードしています: {model_file}")
        
        # 重みをロード (weights_only=Falseでモデル全体をロード)
        checkpoint = torch.load(str(model_file), map_location="cpu", weights_only=False)
        
        # WaveNetworkモデルをインポートして初期化
        from slm.modules.wave_network import WaveNetworkLM
        
        # デフォルトパラメータでモデル初期化
        model = WaveNetworkLM()
        print("モデルを初期化しました")
        
        # 重みを読み込む
        model.load_state_dict(checkpoint, strict=False)
        print("モデル重みをロードしました")
        
        # デバイスに転送
        model.to(device)
        model.eval()  # 評価モードに設定
        
        return model
        
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {str(e)}")
        return None


def prepare_environment(
    data_path: str = "/content/drive/MyDrive/slm/data/fujiki/wiki40b_ja",
    model_path: str = "/content/drive/MyDrive/slm_outputs/slm_1024h_3l/checkpoints/diffusion_step_150.pt",
    target_path: str = "/content/fast_data/"
) -> Dict[str, Any]:
    """
    評価環境を準備します。データコピー、トークナイザーとモデルのロードを行います。
    
    Args:
        data_path: データのパス
        model_path: モデル重みファイルのパス
        target_path: コピー先のパス
        
    Returns:
        Dict[str, Any]: 環境設定の結果情報
    """
    result = {}
    
    # デバイスの選択
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result["device"] = device
    print(f"使用デバイス: {device}")
    
    # データをコピー
    copied_data_path = copy_data_to_fast_storage(data_path, target_path)
    result["data_path"] = copied_data_path
    
    # トークナイザーをロード
    tokenizer_path = os.path.join(copied_data_path, "tokenizers")
    tokenizer = load_tokenizer(tokenizer_path, add_mask_token=True)
    result["tokenizer"] = tokenizer
    
    # モデルをコピー
    copied_model_path = copy_data_to_fast_storage(model_path, target_path)
    result["model_path"] = copied_model_path
    
    # モデルをロード
    model = load_model(copied_model_path, device)
    result["model"] = model
    
    return result


if __name__ == "__main__":
    try:
        # 評価環境を準備
        env = prepare_environment()
        
        print("\n=== 評価環境の準備が完了しました ===")
        print(f"データパス: {env.get('data_path', 'N/A')}")
        print(f"モデルパス: {env.get('model_path', 'N/A')}")
        print(f"トークナイザー: {'ロード成功' if env.get('tokenizer') is not None else 'ロード失敗'}")
        print(f"モデル: {'ロード成功' if env.get('model') is not None else 'ロード失敗'}")
        
        # モデル情報の表示
        if env.get('model') is not None:
            model = env.get('model')
            print("\n=== モデル情報 ===")
            print(f"モデルタイプ: {type(model).__name__}")
            
            if hasattr(model, "embedding_dim"):
                print(f"埋め込み次元: {model.embedding_dim}")
            
            # パラメータ数のカウント
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"総パラメータ数: {total_params:,}")
            print(f"学習可能パラメータ数: {trainable_params:,}")
    
    except Exception as e:
        print(f"スクリプト実行中にエラーが発生しました: {str(e)}")
