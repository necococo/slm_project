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
    add_mask_token: bool = True,
    base_vocab_size: int = 32000
) -> Tuple[Optional[PreTrainedTokenizer], Optional[int]]:
    """
    トークナイザーをロードします。
    
    Args:
        tokenizer_path: トークナイザーが保存されているパス
        use_fast: 高速バージョンを使用するかどうか
        add_mask_token: マスクトークンがない場合に追加するかどうか
        base_vocab_size: 基本語彙サイズ（この範囲内で最後のトークンをマスクトークンと入れ替え）
        
    Returns:
        Tuple[Optional[PreTrainedTokenizer], Optional[int]]: 
            (トークナイザー, マスクトークンID)のタプル
    
    Why not:
        語彙サイズを維持するため、追加ではなく最後のトークンとの入れ替えを行います。
        これにより、モデルの出力層のサイズ変更なしでマスクトークンを利用できます。
    """
    tokenizer_path = Path(tokenizer_path)
    
    if not tokenizer_path.exists():
        print(f"トークナイザーのパスが見つかりません: {tokenizer_path}")
        return None, None
    
    try:
        print(f"トークナイザーをロードしています: {tokenizer_path}")
        
        # AutoTokenizerを使ってトークナイザーをロード
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            use_fast=use_fast
        )
        
        print(f"トークナイザーが正常にロードされました")
        original_vocab_size = len(tokenizer)
        print(f"現在の語彙サイズ: {original_vocab_size}")
        
        # マスクトークンが存在するかチェック
        has_mask_token = hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None
        mask_token_id = None
        
        if not has_mask_token and add_mask_token:
            # 語彙サイズを維持するため、最後のトークンをマスクトークンと入れ替え
            last_token_id = min(base_vocab_size - 1, original_vocab_size - 1)
            
            # 元の最後のトークンの情報を取得
            last_token = tokenizer.convert_ids_to_tokens(last_token_id)
            print(f"最後のトークン(ID={last_token_id}): '{last_token}'")
            
            # マスクトークンを追加
            mask_token = "<mask>"
            
            # トークナイザーの語彙を直接更新
            # これはトークナイザーの実装に依存するため、代替手段も提供
            try:
                # 方法1: トークン名を直接置き換え（もっとも安全な方法）
                if hasattr(tokenizer, "vocab"):
                    # 最後のトークンを削除し、マスクトークンを同じIDに割り当て
                    if last_token in tokenizer.vocab:
                        del tokenizer.vocab[last_token]
                    tokenizer.vocab[mask_token] = last_token_id
                    
                    # 逆マッピング（ID→トークン）も更新
                    if hasattr(tokenizer, "ids_to_tokens"):
                        tokenizer.ids_to_tokens[last_token_id] = mask_token
                    
                    # マスクトークンのプロパティを設定
                    tokenizer.mask_token = mask_token
                    tokenizer.mask_token_id = last_token_id
                    
                    print(f"最後のトークンをマスクトークンに置き換えました。マスクトークンID: {last_token_id}")
                
                # 方法2: 特殊トークンAPIを使用
                else:
                    # この方法は新しいトークンを追加するため、語彙サイズが増えることがある
                    # エラーが発生した場合のフォールバックとしてのみ使用
                    special_tokens_dict = {'mask_token': mask_token}
                    num_added = tokenizer.add_special_tokens(special_tokens_dict)
                    mask_token_id = tokenizer.mask_token_id
                    print(f"特殊トークンAPIでマスクトークンを追加しました（注意: 語彙サイズが{num_added}増加）")
                    print(f"マスクトークンID: {mask_token_id}")
            
            except Exception as e:
                print(f"マスクトークン置換中にエラーが発生: {e}")
                print("代替手段としてトークナイザーの特殊トークンAPIを使用します")
                
                special_tokens_dict = {'mask_token': mask_token}
                tokenizer.add_special_tokens(special_tokens_dict)
                mask_token_id = tokenizer.mask_token_id
                print(f"マスクトークンを追加しました。マスクトークンID: {mask_token_id}")
        
        elif has_mask_token:
            mask_token_id = tokenizer.mask_token_id
            print(f"既存のマスクトークン: {tokenizer.mask_token} (ID: {mask_token_id})")
        
        final_vocab_size = len(tokenizer)
        print(f"最終語彙サイズ: {final_vocab_size}")
        print(f"語彙サイズの変化: {'+' if final_vocab_size > original_vocab_size else ''}{final_vocab_size - original_vocab_size}")
        
        return tokenizer, mask_token_id
    
    except Exception as e:
        print(f"トークナイザーのロード中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def load_model(
    model_path: str,
    vocab_size: int = 32000,
    mask_token_id: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Optional[Any]:
    """
    学習済みモデルを読み込みます。
    
    Args:
        model_path: モデル重みファイルのパス
        vocab_size: モデルの語彙サイズ (マスクトークンを含まない)
        mask_token_id: マスクトークンID (存在する場合)
        device: モデルをロードするデバイス
        
    Returns:
        Optional[Any]: ロードされたモデル
    
    How:
        モデル重みをロードし、必要な設定でWaveNetworkモデルを初期化します。
        基本語彙サイズは32000としつつ、マスクトークンが追加された場合はそれを考慮します。
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
        
        # 重みをロード
        checkpoint = torch.load(str(model_file), map_location=device, weights_only=False)
        
        # WaveNetworkモデルをインポートして初期化
        from slm.modules.wave_network import WaveNetworkLM
        from types import SimpleNamespace
        
        # 必要な設定を作成
        # Why not: 語彙サイズは固定サイズを使用し、
        # マスクトークンは既存の語彙内のトークンと入れ替えているため、
        # 語彙サイズを増やす必要がありません
        config = SimpleNamespace(
            vocab_size=vocab_size,    # 基本語彙サイズを固定で使用
            hidden_size=1024,         # 隠れ層サイズ
            num_layers=3,             # レイヤー数
            max_seq_len=512,          # 最大シーケンス長
            dropout_prob=0.1,         # ドロップアウト確率
            use_rope=True             # 回転位置エンコーディングの使用
        )
        
        # configを渡してモデルを初期化
        model = WaveNetworkLM(config)
        print("モデルを初期化しました")
        print(f"モデル設定: hidden_size={config.hidden_size}, "
              f"num_layers={config.num_layers}, vocab_size={config.vocab_size}, "
              f"max_seq_len={config.max_seq_len}, use_rope={config.use_rope}")
        
        # 重みを読み込む
        model.load_state_dict(checkpoint, strict=False)
        print("モデル重みをロードしました")
        
        # デバイスに転送
        model.to(device)
        model.eval()  # 評価モードに設定
        
        return model
        
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()  # デバッグ情報を出力
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
    
    # 基本語彙サイズ = 32000
    base_vocab_size = 32000
    
    # トークナイザーをロード（基本語彙サイズを渡す）
    tokenizer_path = os.path.join(copied_data_path, "tokenizers")
    tokenizer, mask_token_id = load_tokenizer(
        tokenizer_path, 
        add_mask_token=True, 
        base_vocab_size=base_vocab_size
    )
    result["tokenizer"] = tokenizer
    
    # モデルをコピー
    copied_model_path = copy_data_to_fast_storage(model_path, target_path)
    result["model_path"] = copied_model_path
    
    # モデルをロード（マスクトークンIDと基本語彙サイズを渡す）
    model = load_model(
        copied_model_path, 
        vocab_size=base_vocab_size,
        mask_token_id=mask_token_id,
        device=device
    )
    result["model"] = model
    
    # コレーションユーティリティを準備（マスクトークンIDを明示的に渡す）
    if tokenizer is not None and model is not None:
        from types import SimpleNamespace
        model_config = SimpleNamespace(max_seq_len=512)
        
        try:
            from slm.collator import CustomCollator
            collator = CustomCollator(
                tokenizer=tokenizer,
                model_config=model_config,
                mlm=True,
                mlm_probability=0.15,
                mask_token_id=mask_token_id
            )
            result["collator"] = collator
            print("評価用のCollatorを準備しました")
        except ImportError:
            print("Collatorのインポートに失敗しました")
    
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
