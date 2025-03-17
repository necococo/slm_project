#!/usr/bin/env python
# coding: utf-8
# slm/run_eval.py - 保存されたモデルの評価を実行するスクリプト
# Why: チェックポイントから重みを読み込み、複数の評価指標で言語モデルの性能を測定する

import os
import argparse
import torch
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from typing import Dict, Any, List, Optional

from slm.config import ModelConfig
from slm.modules.wave_network import WaveNetworkLM
from slm.evaluation import (
    evaluate_perplexity, 
    evaluate_bleu, 
    evaluate_rouge,
    temperature_sampling_decode
)
from slm.tokenizer import load_tokenizer


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description="保存済みのSLMモデルの評価を実行します")
    
    # モデル・データ関連
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="評価するモデルチェックポイントのパス (例: ./outputs/checkpoints/final_model.pt)")
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="評価用データセットのパス (HuggingFaceのデータセット名またはディレクトリパス)")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="トークナイザーのパス (指定しない場合、チェックポイントから読み込み)")
    parser.add_argument("--tokenizer_name", type=str, default="megagonlabs/t5-base-japanese-web",
                        help="トークナイザー名 (tokenizer_pathが指定されていない場合に使用)")
    
    # 評価設定
    parser.add_argument("--batch_size", type=int, default=8,
                        help="評価時のバッチサイズ")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="生成するトークンの最大数")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="サンプリング温度 (1.0で等確率、小さいほどgreedy)")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="評価用サンプル数 (0の場合はデータセット全体を使用)")
    
    # 出力関連
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="評価結果の出力ディレクトリ")
    parser.add_argument("--generate_samples", type=int, default=5,
                        help="生成サンプル数 (0の場合は生成しない)")
    parser.add_argument("--device", type=str, default=None,
                        help="使用するデバイス (例: 'cuda', 'cpu', 'mps')")

    args = parser.parse_args()
    
    # デバイス自動検出
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    
    return args


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda") -> tuple[WaveNetworkLM, Any]:
    """
    How:
        チェックポイントからモデルとトークナイザーを読み込む
        
    Args:
        checkpoint_path: モデルチェックポイントのパス
        device: モデルを配置するデバイス
        
    Returns:
        モデルとトークナイザーのタプル
    """
    print(f"チェックポイント {checkpoint_path} からモデルを読み込み中...")
    
    # PyTorch 2.6+対応: 必要なクラスを安全なグローバルとして明示的に登録
    try:
        import torch.serialization
        from slm.config import ModelConfig, TrainingConfig
        
        # ModelConfigとTrainingConfigを安全なグローバルとして登録
        torch.serialization.add_safe_globals([ModelConfig, TrainingConfig])
        
        # T5Tokenizerも安全なグローバルとして登録（T5Tokenizerはチェックポイントに含まれている場合がある）
        try:
            from transformers.models.t5.tokenization_t5 import T5Tokenizer
            torch.serialization.add_safe_globals([T5Tokenizer])
        except ImportError:
            pass  # transformersがインストールされていない場合は無視
            
        # トランスフォーマーの他の可能性のあるクラスも登録
        try:
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase
            torch.serialization.add_safe_globals([PreTrainedTokenizerBase])
        except ImportError:
            pass
            
        print("PyTorch 2.6+用の安全なグローバル登録が完了しました")
    except (ImportError, AttributeError) as e:
        # 古いPyTorchバージョンでは無視
        print(f"PyTorch 2.6+の機能が見つかりません。通常のロード方式を使用します: {e}")
    
    try:
        # まず weights_only=False で試行（PyTorch 2.6+ではデフォルトでTrue）
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            print("weights_only=False でチェックポイントを読み込みました")
        except (TypeError, AttributeError):
            # 古いPyTorchバージョンではweights_onlyがない
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            print("標準モードでチェックポイントを読み込みました")
        
    except Exception as e:
        print(f"チェックポイントの読み込みエラー: {e}")
        print(f"メタデータのみの読み込みを試みます...")
        
        # 部分的に読み込み可能なケースの対応
        import pickle
        try:
            with open(checkpoint_path, 'rb') as f:
                # ヘッダー部分だけを読み込む試み
                header = pickle.load(f)
                if isinstance(header, dict) and "model_config" in header:
                    checkpoint = {"model_config": header["model_config"]}
                    print(f"チェックポイントのメタデータ読み込みに成功しました")
                else:
                    raise ValueError("無効なチェックポイント形式")
        except Exception as e2:
            print(f"メタデータ読み込み失敗: {e2}")
            raise ValueError(f"チェックポイント {checkpoint_path} の読み込みに失敗しました") from e
        
    # モデル設定を取得
    if "model_config" not in checkpoint:
        raise ValueError(f"チェックポイントにmodel_configが見つかりません: {checkpoint_path}")
    
    # チェックポイントのマップを検証
    model_config = checkpoint["model_config"]
    
    # ModelConfigの属性を検証し、必要に応じて修復
    # PyTorch 2.6+でのセキュリティ強化により内部属性が変更されている可能性がある
    if not hasattr(model_config, 'hidden_size') and hasattr(model_config, '_hidden_size'):
        # _属性から通常属性に変換
        for attr_name in dir(model_config):
            if attr_name.startswith('_') and not attr_name.startswith('__'):
                # プライベート属性名（_で始まる）から公開属性名に変換
                public_name = attr_name[1:] # _をスキップ
                if not hasattr(model_config, public_name):
                    # 公開属性を作成
                    value = getattr(model_config, attr_name)
                    setattr(model_config, public_name, value)
                    print(f"属性を修復: {attr_name} → {public_name}")
    
    # 最終検証
    if not hasattr(model_config, 'hidden_size'):
        # モデル設定のデバッグ情報を表示
        print(f"警告: モデル設定に 'hidden_size' 属性が見つかりません")
        print(f"利用可能な属性: {dir(model_config)}")
        
        # 新しいModelConfigオブジェクトを作成
        from slm.config import ModelConfig
        fixed_config = ModelConfig(
            hidden_size=1024,
            num_layers=12,
            vocab_size=32000,
            max_seq_len=512
        )
        
        # 可能な限り元の設定から値をコピー
        for attr in ['hidden_size', 'num_layers', 'vocab_size', 'max_seq_len', 
                     'dropout_prob', 'use_rope', 'use_wavelet', 'wavelet_name',
                     'activation', 'use_bio_noise', 'noise_std']:
            if hasattr(model_config, attr):
                setattr(fixed_config, attr, getattr(model_config, attr))
                
        # 修正された設定を使用
        model_config = fixed_config
        print("モデル設定を修復しました")
            
    print(f"モデル設定: hidden_size={model_config.hidden_size}, num_layers={model_config.num_layers}, vocab_size={model_config.vocab_size}")
    
    # モデルの初期化
    model = WaveNetworkLM(model_config)
    
    # 重みの読み込み
    try:
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("モデルの重みを正常に読み込みました")
        else:
            print("警告: チェックポイントにmodel_state_dictが見つかりません")
    except Exception as e:
        print(f"モデルの重み読み込みエラー: {e}")
        
    # モデルをデバイスに移動
    model = model.to(device)
    model.eval()  # 評価モード
    
    # トークナイザーの取得 - チェックポイントから直接ロードせずに設定のみを確認
    tokenizer = None
    try:
        # 設定からtokenizer_nameを取得
        tokenizer_name = None
        if hasattr(model_config, "tokenizer") and hasattr(model_config.tokenizer, "name_or_path"):
            tokenizer_name = model_config.tokenizer.name_or_path
            print(f"チェックポイントからトークナイザー名を取得: {tokenizer_name}")
        
        # トークナイザー名からトークナイザーを直接ロード
        if tokenizer_name:
            print(f"トークナイザー '{tokenizer_name}' を直接ロードします")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            print(f"トークナイザーのロードに成功しました")
        else:
            # tokenizer属性から直接トークナイザーオブジェクトを取得
            if hasattr(model_config, "tokenizer") and model_config.tokenizer is not None:
                try:
                    # トークナイザーオブジェクトが直接保存されていた場合
                    tokenizer = model_config.tokenizer
                    print(f"モデル設定から直接トークナイザーを取得しました")
                except Exception as e:
                    print(f"トークナイザー取得エラー: {e}")
            else:
                print("モデル設定からトークナイザー情報が取得できませんでした")
    except Exception as e:
        print(f"トークナイザーのロードエラー: {e}")
        
    # トークナイザーが取得できなかった場合はデフォルトトークナイザーを使用
    if tokenizer is None:
        print("デフォルトの日本語トークナイザーをロードします")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("megagonlabs/t5-base-japanese-web")
            print("デフォルトトークナイザーのロードに成功しました")
        except Exception as e2:
            print(f"デフォルトトークナイザーのロードにも失敗: {e2}")
    
    return model, tokenizer


def prepare_test_dataset(args):
    """
    How:
        評価用データセットを準備する
        
    Why not:
        初回実行時はローカルデータがないため、自動的にコピーするようにする
    """
    # データパスが指定されていない場合、デフォルトでwikitext-103を使用
    if args.test_data_path is None:
        print("テストデータセットが指定されていないため、wikitext-103からサンプルをロードします")
        try:
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
            if args.n_samples > 0:
                dataset = dataset.select(range(min(args.n_samples, len(dataset))))
            return dataset
        except Exception as e:
            print(f"wikitext-103のロードに失敗しました: {e}")
            print("代替データソースを試みます...")
    
    # ローカルディレクトリからロード
    if os.path.isdir(args.test_data_path):
        print(f"ディレクトリ {args.test_data_path} からテストデータセットをロード中...")
        
        # 指定されたパスがまだ存在しない場合は、代替パスを確認
        if not os.path.exists(os.path.join(args.test_data_path, "dataset_info.json")) and not os.path.exists(os.path.join(args.test_data_path, "data")):
            print(f"警告: {args.test_data_path} に有効なデータセットが見つかりません")
            
            # Google Driveなど別の場所からコピーする可能性のあるパスを探す
            potential_paths = [
                f"/content/drive/MyDrive/slm/data/fujiki/wiki40b_ja/test",
                f"/content/drive/MyDrive/slm/test",
                f"./data/fujiki/wiki40b_ja/test",
                f"./data/test",
                f"/content/data/test"
            ]
            
            src_path = None
            for path in potential_paths:
                if os.path.exists(path) and (
                    os.path.exists(os.path.join(path, "dataset_info.json")) or 
                    os.path.exists(os.path.join(path, "data"))
                ):
                    src_path = path
                    break
            
            # コピー元が見つかった場合
            if src_path:
                print(f"代替データソースを発見しました: {src_path}")
                print(f"テストデータを {src_path} から {args.test_data_path} にコピーします")
                
                try:
                    # ディレクトリ作成
                    os.makedirs(args.test_data_dir, exist_ok=True)
                    
                    # データをコピー
                    import shutil
                    shutil.copytree(src_path, args.test_data_path, dirs_exist_ok=True)
                    print(f"データコピー完了: {src_path} → {args.test_data_path}")
                except Exception as e:
                    print(f"データコピーエラー: {e}")
                    print("Hugging Faceからのデータ取得に切り替えます...")
                    return load_default_dataset(args)
        
        try:
            dataset = load_from_disk(args.test_data_path)
            if isinstance(dataset, dict) and "test" in dataset:
                dataset = dataset["test"]
            print(f"データセットをロードしました: {len(dataset)} サンプル")
        except Exception as e:
            print(f"ローカルデータセットのロードエラー: {e}")
            print("Hugging Faceからのデータ取得に切り替えます...")
            return load_default_dataset(args)
    # Hugging Faceからロード
    elif args.test_data_path and not os.path.isdir(args.test_data_path):
        print(f"Hugging Face から {args.test_data_path} をロード中...")
        try:
            dataset = load_dataset(args.test_data_path, split="test")
            print(f"データセットをロードしました: {len(dataset)} サンプル")
        except Exception as e:
            print(f"Hugging Face データセットのロードエラー: {e}")
            print("デフォルトデータセットにフォールバックします...")
            return load_default_dataset(args)
    else:
        print("有効なデータパスが指定されていません。デフォルトデータセットを使用します。")
        return load_default_dataset(args)
    
    # サンプル数の制限
    if args.n_samples > 0 and len(dataset) > args.n_samples:
        dataset = dataset.select(range(args.n_samples))
        print(f"データセットを {args.n_samples} サンプルに制限しました")
    
    return dataset

def load_default_dataset(args):
    """
    How:
        デフォルトのテストデータセットをロードする（フォールバック用）
        
    Args:
        args: コマンドライン引数
        
    Returns:
        データセットオブジェクト
    """
    print("デフォルトのwikitext-103データセットをロード中...")
    try:
        # wikitextデータセットのロード
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        
        # サンプル数制限
        if args.n_samples > 0:
            dataset = dataset.select(range(min(args.n_samples, len(dataset))))
            
        print(f"デフォルトデータセットをロードしました: {len(dataset)} サンプル")
        return dataset
    except Exception as e:
        print(f"デフォルトデータセットのロードに失敗しました: {e}")
        print("簡易データセットを生成します...")
        
        # 最小限のデータセットを生成（完全なフォールバック）
        from datasets import Dataset
        minimal_data = {
            "text": [
                "これはテストサンプルです。言語モデルの評価に使用されます。",
                "Wave Networkは小規模でありながら効果的な言語モデルです。",
                "複素ベクトル表現により、少ないパラメータ数で高い表現力を実現しています。"
            ]
        }
        return Dataset.from_dict(minimal_data)


def prepare_tokenizer(args, model_tokenizer=None):
    """
    How:
        トークナイザーを準備する
    """
    if model_tokenizer is not None:
        print("モデルからトークナイザーを取得しました")
        return model_tokenizer
        
    if args.tokenizer_path:
        print(f"トークナイザーを {args.tokenizer_path} からロード中...")
        try:
            tokenizer = load_tokenizer(
                model_name=args.tokenizer_name, 
                tokenizer_path=args.tokenizer_path
            )
        except Exception as e:
            print(f"トークナイザー読み込みエラー: {e}")
            print(f"代わりにトークナイザー {args.tokenizer_name} をダウンロードします")
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        print(f"トークナイザー {args.tokenizer_name} をダウンロード中...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        
    return tokenizer


class TestDatasetWrapper(torch.utils.data.Dataset):
    """
    How:
        テストデータセットを評価用に変換するラッパークラス
    """
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # データフォーマットの確認
        if len(dataset) > 0:
            first_item = dataset[0]
            self.keys = list(first_item.keys()) if isinstance(first_item, dict) else []
            print(f"データセットキー: {self.keys}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # データ形式に応じて変換
        if isinstance(item, dict):
            # 入力テキストを取得 (使用優先順位)
            text = None
            for key in ["text", "content", "article", "input_text"]:
                if key in item:
                    text = item[key]
                    break
                    
            if text is None and len(item) > 0:
                # 最初のキーをテキストとして使用
                text = item[list(item.keys())[0]]
            
            # トークン化
            if self.tokenizer:
                input_ids = self.tokenizer.encode(
                    text, 
                    add_special_tokens=False, 
                    max_length=self.max_length,
                    truncation=True
                )
                
                # 目標テキスト (可能であれば)
                target_text = item.get("target_text", item.get("summary", text))
                
                # 返り値は評価関数が期待する形式に
                return {
                    "input_ids": torch.tensor(input_ids),
                    "labels": torch.tensor(input_ids),  # 自己回帰評価用
                    "text": text,  # 生成サンプル用
                    "target_text": target_text  # BLEU/ROUGE評価用
                }
        
        # 文字列の場合
        elif isinstance(item, str):
            input_ids = self.tokenizer.encode(
                item, 
                add_special_tokens=False, 
                max_length=self.max_length,
                truncation=True
            )
            return {
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(input_ids),
                "text": item,
                "target_text": item
            }
        
        # その他の形式
        return {
            "input_ids": torch.tensor([]),
            "labels": torch.tensor([]),
            "text": "",
            "target_text": ""
        }


def generate_samples(model, dataset, tokenizer, device, max_new_tokens=32, temperature=1.0, num_samples=5):
    """
    How:
        モデルを使用してテキスト生成サンプルを作成
    """
    results = []
    
    # 最大num_samplesまでのサンプルを生成
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        input_ids = sample["input_ids"].to(device).unsqueeze(0)
        
        # 入力テキスト (トークナイザーで戻す)
        if tokenizer and hasattr(tokenizer, 'decode'):
            prefix = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
            # 長すぎる場合は省略
            if len(prefix) > 100:
                prefix = prefix[:100] + "..."
        else:
            # トークナイザーがない場合はテキストをそのまま使用
            prefix = sample["text"][:100] + "..." if len(sample["text"]) > 100 else sample["text"]
        
        # テキスト生成（トークナイザーを渡す）
        with torch.no_grad():
            generated_text = temperature_sampling_decode(
                model, input_ids, max_new_tokens, device, 
                temperature=temperature,
                tokenizer=tokenizer  # トークナイザーを渡す
            )
        
        results.append({
            "prefix": prefix,
            "generated": generated_text
        })
    
    return results


def main():
    """メイン実行関数"""
    # 引数の解析
    args = parse_args()
    
    # デバイスのセットアップ
    device = torch.device(args.device)
    print(f"使用デバイス: {device}")
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # データパスのデフォルト設定を追加（ローカルとドライブのマッピング）
    if args.test_data_path is None:
        # Google Driveがマウントされているか確認
        drive_path = "/content/drive/MyDrive"
        if os.path.exists(drive_path):
            # Googleドライブのデフォルトパス
            default_test_path = os.path.join(drive_path, "slm/data/fujiki/wiki40b_ja/test")
            if os.path.exists(default_test_path):
                print(f"Googleドライブ上のデフォルトテストデータを使用します: {default_test_path}")
                args.test_data_path = default_test_path
            else:
                # ローカルのデフォルトパス
                local_test_path = "/content/fast_data/test"
                if not os.path.exists(local_test_path):
                    os.makedirs(local_test_path, exist_ok=True)
                args.test_data_path = local_test_path
        else:
            # ローカル環境のデフォルトパス
            args.test_data_path = "./data/test"
    
    # モデルのロード - エラー時にフォールバック処理を強化
    try:
        model, model_tokenizer = load_model_from_checkpoint(args.checkpoint_path, str(device))
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")
        print("モデル読み込みに失敗しましたが、処理を継続します。")
        print("新しいモデルを初期化します...")
        
        # 設定ファイルからの読み込みに失敗した場合、デフォルト設定でモデルを初期化
        from slm.config import ModelConfig
        model_config = ModelConfig(
            hidden_size=768,
            num_layers=6,
            vocab_size=32000,
            max_seq_len=512
        )
        model = WaveNetworkLM(model_config)
        model = model.to(device)
        model_tokenizer = None
    
    # トークナイザーの準備 - エラーハンドリングを強化
    try:
        tokenizer = prepare_tokenizer(args, model_tokenizer)
    except Exception as e:
        print(f"トークナイザー準備エラー: {e}")
        print("標準トークナイザーをロードします...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("megagonlabs/t5-base-japanese-web")
        except Exception as e2:
            print(f"標準トークナイザーのロードにも失敗: {e2}")
            print("評価を続行できません。終了します。")
            return
    
    # データセットの準備 - 失敗時のエラーハンドリングを強化
    test_dataset = None
    try:
        test_dataset = prepare_test_dataset(args)
    except Exception as e:
        print(f"データセット準備中にエラーが発生しました: {e}")
        print("内部的なデフォルトデータセットにフォールバックします...")
        test_dataset = load_default_dataset(args)
    
    if test_dataset is None:
        print("テストデータセットの準備に失敗しました。終了します。")
        return
    
    # 評価用にデータセットをラップ
    wrapped_dataset = TestDatasetWrapper(test_dataset, tokenizer, max_length=512)
    print(f"評価用データセットを準備しました: {len(wrapped_dataset)} サンプル")
    
    # パープレキシティ評価
    try:
        ppl = evaluate_perplexity(model, wrapped_dataset, device, batch_size=args.batch_size)
        print(f"Perplexity: {ppl:.4f}")
        
        # 結果の保存
        with open(os.path.join(args.output_dir, "perplexity.txt"), "w") as f:
            f.write(f"Perplexity: {ppl:.4f}\n")
    except Exception as e:
        print(f"Perplexity評価でエラーが発生しました: {e}")
    
    # BLEUスコア評価
    try:
        bleu = evaluate_bleu(
            model, wrapped_dataset, device, 
            batch_size=args.batch_size, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        print(f"BLEU: {bleu:.4f}")
        
        # 結果の保存
        with open(os.path.join(args.output_dir, "bleu.txt"), "w") as f:
            f.write(f"BLEU: {bleu:.4f}\n")
    except Exception as e:
        print(f"BLEU評価でエラーが発生しました: {e}")
    
    # ROUGEスコア評価
    try:
        rouge_results = evaluate_rouge(
            model, wrapped_dataset, device, 
            batch_size=args.batch_size, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        print(f"ROUGE-1: {rouge_results['rouge1']:.4f}")
        print(f"ROUGE-2: {rouge_results['rouge2']:.4f}")
        print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
        
        # 結果の保存
        with open(os.path.join(args.output_dir, "rouge.txt"), "w") as f:
            for k, v in rouge_results.items():
                f.write(f"{k}: {v:.4f}\n")
    except Exception as e:
        print(f"ROUGE評価でエラーが発生しました: {e}")
    
    # テキスト生成サンプル
    if args.generate_samples > 0:
        try:
            samples = generate_samples(
                model, wrapped_dataset, tokenizer, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                num_samples=args.generate_samples
            )
            
            # サンプルの表示と保存
            print("\n=== 生成サンプル ===")
            with open(os.path.join(args.output_dir, "samples.txt"), "w", encoding="utf-8") as f:
                for i, sample in enumerate(samples):
                    print(f"\nサンプル {i+1}:")
                    print(f"入力: {sample['prefix']}")
                    print(f"生成: {sample['generated']}")
                    
                    f.write(f"サンプル {i+1}:\n")
                    f.write(f"入力: {sample['prefix']}\n")
                    f.write(f"生成: {sample['generated']}\n\n")
        except Exception as e:
            print(f"テキスト生成でエラーが発生しました: {e}")
    
    print(f"\n評価結果は {args.output_dir} ディレクトリに保存されました")


if __name__ == "__main__":
    main()
