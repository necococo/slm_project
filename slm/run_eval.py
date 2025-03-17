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
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    学習済みモデルを読み込みます。
    
    Args:
        model_path: モデルが保存されているパス
        device: モデルをロードするデバイス
        
    Returns:
        Tuple[Optional[Any], Optional[Dict[str, Any]]]: (モデル, モデル設定)のタプル
    
    How:
        - モデルの重みを直接PyTorchの形式で読み込む
        - モデル設定ファイルも読み込んで返す
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"モデルのパスが見つかりません: {model_path}")
        return None, None
    
    try:
        print(f"モデルをロードしています: {model_path}")
        
        # モデル設定ファイルの読み込み
        config_path = model_path / "config.pth"
        if config_path.exists():
            config = torch.load(str(config_path), map_location="cpu")
            print(f"モデル設定をロードしました")
        else:
            print(f"モデル設定ファイルが見つかりません: {config_path}")
            config = None
        
        # モデル重みの読み込み
        model_files = list(model_path.glob("*.pth"))
        model_files = [f for f in model_files if f.name != "config.pth"]
        
        if not model_files:
            print(f"モデル重みファイルが見つかりません: {model_path}")
            return None, config
        
        # 最新のモデルをロード（複数ある場合）
        model_file = sorted(model_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
        
        # WaveNetworkモデルをインポートして初期化
        try:
            from slm.modules.wave_network import WaveNetworkLM
            
            if config is not None:
                # 設定からモデルを初期化
                model = WaveNetworkLM(config)
                print(f"設定からWaveNetworkモデルを初期化しました")
            else:
                print("警告: 設定ファイルがないため、デフォルト設定でモデルを初期化します")
                # デフォルト設定でモデルを初期化（必要に応じてパラメータを調整）
                model = WaveNetworkLM()
            
            # 重みの読み込み
            state_dict = torch.load(str(model_file), map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"モデル重みをロードしました: {model_file.name}")
            
            # デバイスに転送
            model.to(device)
            model.eval()  # 評価モードに設定
            
            return model, config
            
        except ImportError:
            print("WaveNetworkモデルをインポートできませんでした。slm.modules.wave_networkが利用可能か確認してください。")
            return None, config
        except Exception as e:
            print(f"モデルのロード中にエラーが発生しました: {str(e)}")
            return None, config
    
    except Exception as e:
        print(f"モデルのロード処理でエラーが発生しました: {str(e)}")
        return None, None


def prepare_environment(
    data_path: str = "/content/drive/MyDrive/slm/data/fujiki/wiki40b_ja",
    model_path: str = "/content/drive/MyDrive/slm_outputs/slm_1024h_3l/checkpoints/diffusion_step_150.pt",
    target_path: str = "/content/fast_data/"
) -> Dict[str, Any]:
    """
    評価環境を準備します。データコピー、トークナイザーとモデルのロードを行います。
    
    Args:
        data_path: データのパス
        model_path: モデルのパス
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
    model, config = load_model(copied_model_path, device)
    result["model"] = model
    result["model_config"] = config
    
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
            config = env.get('model_config', {})
            
            print("\n=== モデル情報 ===")
            print(f"モデルタイプ: {type(model).__name__}")
            
            if hasattr(model, "embedding_dim"):
                print(f"埋め込み次元: {model.embedding_dim}")
            elif config and "embedding_dim" in config:
                print(f"埋め込み次元: {config['embedding_dim']}")
            
            # パラメータ数のカウント
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"総パラメータ数: {total_params:,}")
            print(f"学習可能パラメータ数: {trainable_params:,}")
    
    except Exception as e:
        print(f"スクリプト実行中にエラーが発生しました: {str(e)}")
