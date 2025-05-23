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
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import pyarrow as pa  # Arrowファイル読み込み用


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
            # トークナイザーの実際の語彙サイズを取得し、base_vocab_sizeを上書き
            # Why not: 実際のモデルパラメータに合わせる必要があるため
            actual_vocab_size = len(tokenizer)
            if actual_vocab_size != base_vocab_size:
                print(f"警告: トークナイザーの語彙サイズ({actual_vocab_size})がベース語彙サイズ({base_vocab_size})と異なります")
                # モデルの語彙サイズ以下でトークナイザーの語彙サイズ以下の値に調整
                base_vocab_size = min(base_vocab_size, actual_vocab_size)
                print(f"使用する語彙サイズを {base_vocab_size} に調整します")
            
            # 語彙サイズを維持するため、最後のトークンをマスクトークンと入れ替え
            last_token_id = base_vocab_size - 1
            
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
        # トークナイザーの語彙サイズとモデルの語彙サイズを一致させる
        print(f"モデル初期化時の語彙サイズ: {vocab_size}")
        
        config = SimpleNamespace(
            vocab_size=vocab_size,    # 指定された語彙サイズを使用
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
    target_path: str = "/content/fast_data/",
    vocab_size: int = 32000
) -> Dict[str, Any]:
    """
    評価環境を準備します。データコピー、トークナイザーとモデルのロードを行います。
    
    Args:
        data_path: データのパス
        model_path: モデル重みファイルのパス
        target_path: コピー先のパス
        vocab_size: 基本語彙サイズ
        
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
    
    # トークナイザーをロード（指定された語彙サイズを渡す）
    tokenizer_path = os.path.join(copied_data_path, "tokenizers")
    tokenizer, mask_token_id = load_tokenizer(
        tokenizer_path, 
        add_mask_token=True, 
        base_vocab_size=vocab_size
    )
    result["tokenizer"] = tokenizer
    
    # トークナイザーがロードできた場合、実際の語彙サイズをチェックして調整
    actual_vocab_size = vocab_size
    if tokenizer is not None:
        # マスクトークンが最後の位置にある場合は元の語彙サイズを使用
        if mask_token_id is not None and mask_token_id == vocab_size - 1:
            actual_vocab_size = vocab_size
        else:
            # それ以外の場合はトークナイザーの語彙サイズに合わせる
            actual_vocab_size = min(len(tokenizer), vocab_size)
        
        print(f"モデル初期化に使用する語彙サイズ: {actual_vocab_size}")
    
    # モデルをコピー
    copied_model_path = copy_data_to_fast_storage(model_path, target_path)
    result["model_path"] = copied_model_path
    
    # モデルをロード（調整した語彙サイズを渡す）
    model = load_model(
        copied_model_path, 
        vocab_size=actual_vocab_size,
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


def load_arrow_data(
    arrow_path: str, 
    limit: int = 100
) -> list:
    """
    Arrowファイルからデータを読み込みます。
    
    Args:
        arrow_path: Arrowファイルのパス
        limit: 読み込むサンプル数の上限
        
    Returns:
        list: 読み込んだテキストのリスト
    
    How:
        PyArrowライブラリを使用してArrowファイルを読み込み、
        テキストデータを抽出します。
    """
    print(f"Arrowファイルを読み込んでいます: {arrow_path}")
    try:
        # Arrowファイルを開く
        reader = pa.ipc.open_file(arrow_path)
        table = reader.read_all()
        
        # テキストカラムを取得（通常は'text'という名前）
        text_column = None
        for col_name in table.column_names:
            if (col_name.lower() == 'text' or 'content' in col_name.lower()):):
                text_column = col_name
                break
        
        if text_column is None:
            # カラム名がテキストを表していそうなものを探す
            print(f"利用可能なカラム: {table.column_names}")
            if len(table.column_names) > 0:
                text_column = table.column_names[0]  # 最初のカラムを使用
                print(f"最初のカラム '{text_column}' をテキストとして使用します")
            else:
                raise ValueError("テキストカラムが見つかりません")
        
        # テキストデータを取得
        texts = table[text_column].to_pylist()
        print(f"{len(texts)} サンプルを読み込みました")
        
        # サンプル数を制限
        if limit and limit < len(texts):
            texts = texts[:limit]
            print(f"サンプル数を {limit} に制限しました")
        
        return texts
    
    except Exception as e:
        print(f"Arrowファイルの読み込み中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def evaluate_model(
    model: Any,
    tokenizer: PreTrainedTokenizer,
    collator: Any,
    test_data: Optional[str] = None,
    num_examples: int = 10,7,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
    max_length: int = 512h.device] = None,
) -> Dict[str, Any]:ol = True
    """r:
    モデルを評価し、パープレキシティやサンプル生成を行います。
    モデルによるテキスト生成を行います。
    Args:
        model: 評価対象のモデル
        tokenizer: トークナイザー
        collator: データコレーターコードを行うトークナイザー
        test_data: テストデータのパス (Noneの場合はサンプル文のみ評価)
        num_examples: 生成するサンプル例の数ン数
        batch_size: バッチサイズ性を制御する温度パラメータ
        device: 実行デバイス考慮する上位k個のトークン
        max_length: 最大シーケンス長みを考慮するnucleusサンプリング閾値
        device: 計算を行うデバイス
    Returns:n_output: 特殊文字や制御文字を取り除くか
        Dict[str, Any]: 評価結果を含む辞書
    Returns:
    How:str: 生成されたテキスト
        - サンプルプロンプトからテキスト生成を行う
        - テストデータから損失とパープレキシティを計算する
        - マスク単語予測の精度を評価する、モデルで自己回帰的に次トークンを
    """ 予測しながら文章を生成していきます。生成の多様性と品質のバランスを
    if device is None:_pサンプリングを組み合わせています。
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device is None:
    model.eval()  # 評価モードに設定e("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    # 入力のエンコード
    # サンプルプロンプトの定義kenizer.encode(prompt, return_tensors="pt").to(device)
    sample_prompts = [orch.ones_like(input_ids)
        "こんにちは、私の名前は",
        "今日の天気は",
        "日本の首都は",= input_ids.shape[1]
        "人工知能の研究は",
        "大規模言語モデルとは"
    ]odel.eval()
    with torch.no_grad():
    # テキスト生成による評価range(max_new_tokens):
    print("\n=== テキスト生成評価 ===")
    generated_texts = []del(input_ids, attention_mask=attention_mask)
            next_token_logits = outputs[:, -1, :]
    for prompt in sample_prompts[:num_examples]:
        print(f"\nプロンプト: {prompt}")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
        # 生成パラメータ-K サンプリング
        gen_length = min(max_length - input_ids.shape[1], 50)  # 生成する最大トークン数
                indices_to_remove = torch.topk(scaled_logits, k=top_k)[0][..., -1, None] > scaled_logits
        # 実際の生成処理caled_logits = scaled_logits.masked_fill(indices_to_remove, float('-inf'))
        with torch.no_grad():
            # 単純な自己回帰生成cleus) サンプリング
            for _ in range(gen_length):
                # フォワードパスgits, sorted_indices = torch.sort(scaled_logits, descending=True)
                outputs = model(input_ids)umsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                next_token_logits = outputs[:, -1, :]
                # 累積確率がtop_pを超える位置を特定
                # 次のトークンを選択（温度ありのサンプリング） = cumulative_probs > top_p
                temperature = 0.7
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                # シフトして次の位置からマスク
                # 生成シーケンスを拡張es_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # EOSトークンが生成されたら終了
                if next_token.item() == tokenizer.eos_token_id:its, dtype=torch.bool).scatter_(
                    breakorted_indices, sorted_indices_to_remove
                )
        # 生成されたテキストをデコードogits = scaled_logits.masked_fill(indices_to_remove, float('-inf'))
        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"生成結果: {generated_text}")
        generated_texts.append({"prompt": prompt, "generated": generated_text})
            next_token = torch.multinomial(probs, num_samples=1)
    results["generated_texts"] = generated_texts
            # 入力シーケンスの拡張
    # テストデータがある場合、パープレキシティを計算.cat([input_ids, next_token], dim=1)
    if test_data is not None:torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
        try:
            print("\n=== パープレキシティ評価 ===")
            print(f"テストデータパス: {test_data}")t None and next_token.item() == tokenizer.eos_token_id:
                break
            # ファイル形式に応じたデータ読み込み処理
            test_texts = []分を含む）
            if test_data.endswith('.arrow'):], skip_special_tokens=True)
                # Arrowファイルからデータを読み込み
                test_texts = load_arrow_data(test_data, limit=100)
            elif test_data.endswith('.jsonl'):f full_text.startswith(prompt) else full_text
                # JSONLファイルからデータを読み込み
                import json
                with open(test_data, "r", encoding="utf-8") as f:
                    test_texts = [json.loads(line)["text"] for line in f][:100]
            else:御文字を除去
                # テキストファイルとして処理(r'[\x00-\x1F\x7F-\x9F]', '', generated_text)
                with open(test_data, "r", encoding="utf-8") as f:
                    test_texts = [line.strip() for line in f][:100]ext)
            
            print(f"テストデータを{len(test_texts)}サンプルロードしました")
            
            # データの前処理
            from torch.utils.data import Dataset, DataLoader
            ny,
            class SimpleDataset(Dataset):
                def __init__(self, texts, tokenizer, max_length):
                    self.examples = []
                    for text in texts:
                        encoding = tokenizer(
                            text,] = None,
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt"
                        )生成を行います。
                        self.examples.append({
                            "input_ids": encoding["input_ids"][0],
                            "attention_mask": encoding["attention_mask"][0]
                        })
        collator: データコレーター
        test_data: テストデータのパス (Noneの場合はサンプル文のみ評価)
        num_examples: 生成するサンプル例の数
        batch_size: バッチサイズ
        device: 実行デバイス
        max_length: 最大シーケンス長
        
    Returns:
        Dict[str, Any]: 評価結果を含む辞書
    
    How:
        - サンプルプロンプトからテキスト生成を行う
        - テストデータから損失とパープレキシティを計算する
        - マスク単語予測の精度を評価する
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()  # 評価モードに設定
    results = {}
    
    # サンプルプロンプトの定義
    sample_prompts = [
                    labels = batch["labels"].to(device)
        "今日の天気は",
        "日本の首都は",
                    loss = outputs.loss.item()
        "大規模言語モデルとは"
                    # 有効なトークン数（-100でないラベル）を計算
                    valid_tokens = (labels != -100).sum().item()
                    
                    total_loss += loss * valid_tokens
                    total_length += valid_tokens
            
            avg_loss = total_loss / total_length if total_length > 0 else float('inf')
            perplexity = np.exp(avg_loss)
            
            print(f"テストセットのパープレキシティ: {perplexity:.4f}")
            results["perplexity"] = perplexity
            results["loss"] = avg_loss
            tokenizer=tokenizer,
        except Exception as e:
            print(f"パープレキシティ計算中にエラーが発生しました: {e}")
            import traceback  # 少し高めの温度で多様性を増加
            traceback.print_exc()
            top_p=0.9,
    # マスク単語予測の評価ce=device,
    try:    clean_output=True
        print("\n=== マスク単語予測評価 ===")
        mask_examples = [
            f"日本の首都は{tokenizer.mask_token}です。",)
            f"{tokenizer.mask_token}は美しい島国です。",
            f"人工知能の研究は{tokenizer.mask_token}分野で行われています。",
            f"{tokenizer.mask_token}が開発した相対性理論は物理学に革命をもたらした。"
        ])
        
        mask_predictions = []] = generated_texts
        
        for example in mask_examples:
            print(f"\n入力文: {example}")
            inputs = tokenizer(example, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            print(f"テストデータパス: {test_data}")
            # マスク位置を特定
            mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            test_texts = []
            with torch.no_grad():('.arrow'):
                outputs = model(**inputs)
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            elif test_data.endswith('.jsonl'):
            # 各マスク位置で予測ファイルからデータを読み込み
            for pos in mask_positions:
                mask_logits = logits[0, pos, :]ing="utf-8") as f:
                topk = torch.topk(mask_logits, k=5)"text"] for line in f][:100]
                topk_tokens = [tokenizer.decode([idx]) for idx in topk.indices]
                topk_probs = topk.values.softmax(dim=-1).cpu().numpy()
                with open(test_data, "r", encoding="utf-8") as f:
                print(f"予測トップ5: {list(zip(topk_tokens, topk_probs.tolist()))}")
                mask_predictions.append({
                    "text": example,_texts)}サンプルロードしました")
                    "predictions": [{"token": t, "prob": float(p)} for t, p in zip(topk_tokens, topk_probs)]
                })前処理
            from torch.utils.data import Dataset, DataLoader
        results["mask_predictions"] = mask_predictions
            class SimpleDataset(Dataset):
    except Exception as e:__(self, texts, tokenizer, max_length):
        print(f"マスク予測中にエラーが発生しました: {e}")
        import tracebacktext in texts:
        traceback.print_exc()ing = tokenizer(
                            text,
    return results          truncation=True,
                            max_length=max_length,
                            return_tensors="pt"
def debug_model_output(
    model: Any,                 self.examples.append({
    tokenizer: PreTrainedTokenizer, ut_ids": encoding["input_ids"][0],
    input_text: str, ing["attention_mask"][0]
    device: torch.device                })
) -> None:
    """
    モデルの出力をデバッグするためのヘルパー関数です。
    
    Args:
        model: 検査するモデル            return self.examples[idx]
        tokenizer: 使用するトークナイザー
        input_text: 入力テキスト
        device: 計算デバイスaset(test_texts, tokenizer, max_length)
    ader(
    How:
        入力テキストをトークン化し、モデルに通してその出力形式と値を検査します。    batch_size=batch_size, 
        これによりモデルの動作を理解し、適切な処理方法を見つけることができます。
    """
    print(f"\n=== モデル出力デバッグ ({input_text}) ===")
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # 入力トークンIDs
    print(f"入力トークン: {input_ids.tolist()}")
    
    # モデル出力の取得        for batch in tqdm(test_dataloader, desc="パープレキシティ計算"):
    with torch.no_grad():  input_ids = batch["input_ids"].to(device)
        outputs = model(input_ids)
    "].to(device)
    # 出力の型と形状を確認        
    print(f"出力の型: {type(outputs)}")input_ids, attention_mask=attention_mask, labels=labels)
    if isinstance(outputs, torch.Tensor):
        print(f"出力の形状: {outputs.shape}")
        print(f"出力のデータ型: {outputs.dtype}")        # 有効なトークン数（-100でないラベル）を計算
    else:!= -100).sum().item()
        for i, output in enumerate(outputs):
            if isinstance(output, torch.Tensor):
                print(f"出力[{i}]の形状: {output.shape}, データ型: {output.dtype}")
    
    # 最後のトークンの予測確率を確認loss / total_length if total_length > 0 else float('inf')
    if isinstance(outputs, torch.Tensor):
        last_token_logits = outputs[0, -1, :]
    else:")
        # モデル固有の出力構造に応じて調整results["perplexity"] = perplexity
        if hasattr(outputs, "logits"):["loss"] = avg_loss
            last_token_logits = outputs.logits[0, -1, :]
        else:
            last_token_logits = outputs[0][0, -1, :] {e}")
    
    # 上位5つのトークンとその確率を表示
    topk = torch.topk(last_token_logits, k=5)
    probs = F.softmax(topk.values, dim=-1)の評価
    
    print("次トークンの予測（Top 5）:")ク単語予測評価 ===")
    for i, (token_id, prob) in enumerate(zip(topk.indices.tolist(), probs.tolist())):
        token = tokenizer.decode([token_id])。",
        print(f"{i+1}. '{token}' (ID: {token_id}, 確率: {prob:.4f})")
f"人工知能の研究は{tokenizer.mask_token}分野で行われています。",

if __name__ == "__main__":
    try:
        # 評価環境を準備（語彙サイズ32100を指定）tions = []
        env = prepare_environment(vocab_size=32100)
        mask_examples:
        print("\n=== 評価環境の準備が完了しました ===")
        print(f"データパス: {env.get('data_path', 'N/A')}")
        print(f"モデルパス: {env.get('model_path', 'N/A')}")n inputs.items()}
        print(f"トークナイザー: {'ロード成功' if env.get('tokenizer') is not None else 'ロード失敗'}")        
        print(f"モデル: {'ロード成功' if env.get('model') is not None else 'ロード失敗'}")
        "] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        # モデル情報の表示
        if env.get('model') is not None:d():
            model = env.get('model')                outputs = model(**inputs)



































































        traceback.print_exc()        import traceback        print(f"スクリプト実行中にエラーが発生しました: {str(e)}")    except Exception as e:                print(f"評価結果を保存しました: {result_path}")                json.dump(eval_results, f, ensure_ascii=False, indent=2)            with open(result_path, "w", encoding="utf-8") as f:            import json            result_path = "evaluation_results.json"            # 結果の保存                        print(f"マスク予測サンプル数: {len(eval_results.get('mask_predictions', []))}")            print(f"生成サンプル数: {len(eval_results.get('generated_texts', []))}")                            print(f"パープレキシティ: {eval_results['perplexity']:.4f}")            if "perplexity" in eval_results:            print("\n=== 評価結果サマリー ===")            # 結果のサマリー表示                        )                device=env["device"]                test_data=test_data,                collator=env.get("collator"),                tokenizer=env["tokenizer"],                model=env["model"],            eval_results = evaluate_model(            # 評価の実行                            print(f"テストデータが見つかりました: {test_data}")            else:                print("テストデータが見つかりません。サンプル生成のみで評価を行います。")            if not test_data:                            test_data = jsonl_path if os.path.exists(jsonl_path) else None                jsonl_path = os.path.join(env.get("data_path", ""), "validation.jsonl")            if not test_data:            # Arrow形式が見つからない場合はJSONLファイルを探す                        test_data = arrow_path if os.path.exists(arrow_path) else None            arrow_path = os.path.join(env.get("data_path", ""), "test/data-00000-of-00001.arrow")            # テストデータのパス（Arrow形式ファイル）                        print("\n=== モデル評価を開始します ===")                        debug_model_output(model, tokenizer, "こんにちは", device)            # モデルの出力形式を確認                        device = env.get("device")            tokenizer = env.get("tokenizer")             model = env.get("model")            # モデルの出力形式をデバッグ        if env.get("model") is not None and env.get("tokenizer") is not None:        # モデル評価を実行                    print(f"学習可能パラメータ数: {trainable_params:,}")            print(f"総パラメータ数: {total_params:,}")            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)            total_params = sum(p.numel() for p in model.parameters())            # パラメータ数のカウント                            print(f"埋め込み次元: {model.embedding_dim}")            if hasattr(model, "embedding_dim"):                        print(f"モデルタイプ: {type(model).__name__}")            print("\n=== モデル情報 ===")                logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            
            # 各マスク位置で予測
            for pos in mask_positions:
                mask_logits = logits[0, pos, :]
                topk = torch.topk(mask_logits, k=5)
                topk_tokens = [tokenizer.decode([idx]) for idx in topk.indices]
                topk_probs = topk.values.softmax(dim=-1).cpu().numpy()
                
                print(f"予測トップ5: {list(zip(topk_tokens, topk_probs.tolist()))}")
                mask_predictions.append({
                    "text": example,
                    "predictions": [{"token": t, "prob": float(p)} for t, p in zip(topk_tokens, topk_probs)]


















        traceback.print_exc()        import traceback        print(f"スクリプト実行中にエラーが発生しました: {str(e)}")    except Exception as e:                print(f"評価結果を保存しました: {result_path}")                json.dump(eval_results, f, ensure_ascii=False, indent=2)            with open(result_path, "w", encoding="utf-8") as f:            import json            result_path = "evaluation_results.json"            # 結果の保存                        print(f"マスク予測サンプル数: {len(eval_results.get('mask_predictions', []))}")            print(f"生成サンプル数: {len(eval_results.get('generated_texts', []))}")



                            print(f"パープレキシティ: {eval_results['perplexity']:.4f}")


            if "perplexity" in eval_results:            print("\n=== 評価結果サマリー ===")
            # 結果のサマリー表示                        )                device=env["device"]


                test_data=test_data,                collator=env.get("collator"),                tokenizer=env["tokenizer"],


                model=env["model"],            eval_results = evaluate_model(            # 評価の実行
                            print(f"テストデータが見つかりました: {test_data}")            else:

                print("テストデータが見つかりません。サンプル生成のみで評価を行います。")
            if not test_data:                            test_data = jsonl_path if os.path.exists(jsonl_path) else None

                jsonl_path = os.path.join(env.get("data_path", ""), "validation.jsonl")            if not test_data:                })
        



            # Arrow形式が見つからない場合はJSONLファイルを探す                        test_data = arrow_path if os.path.exists(arrow_path) else None        results["mask_predictions"] = mask_predictions
    

            arrow_path = os.path.join(env.get("data_path", ""), "test/data-00000-of-00001.arrow")    except Exception as e:


            # テストデータのパス（Arrow形式ファイル）

                        print("\n=== モデル評価を開始します ===")        if env.get("model") is not None and env.get("tokenizer") is not None:


        # モデル評価を実行
                    print(f"学習可能パラメータ数: {trainable_params:,}")

            print(f"総パラメータ数: {total_params:,}")            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            total_params = sum(p.numel() for p in model.parameters())
            # パラメータ数のカウント                            print(f"埋め込み次元: {model.embedding_dim}")


            if hasattr(model, "embedding_dim"):                        print(f"モデルタイプ: {type(model).__name__}")        print(f"マスク予測中にエラーが発生しました: {e}")


            print("\n=== モデル情報 ===")            model = env.get('model')


        if env.get('model') is not None:        # モデル情報の表示        

        print(f"モデル: {'ロード成功' if env.get('model') is not None else 'ロード失敗'}")

        print(f"トークナイザー: {'ロード成功' if env.get('tokenizer') is not None else 'ロード失敗'}")

        print(f"モデルパス: {env.get('model_path', 'N/A')}")

        print(f"データパス: {env.get('data_path', 'N/A')}")
        print("\n=== 評価環境の準備が完了しました ===")                env = prepare_environment(vocab_size=32100)


        # 評価環境を準備（語彙サイズ32100を指定）


    try:if __name__ == "__main__":    return results            traceback.print_exc()        import traceback