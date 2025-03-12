# slm/tokenizer.py
# SLM用の簡素化されたトークナイザーユーティリティ - megagonlabs/t5-base-japanese-web専用

from transformers import AutoTokenizer
from typing import Dict, Any, List

def load_tokenizer(model_name: str = "megagonlabs/t5-base-japanese-web", tokenizer_path: str = None, 
                save_dir: str = None, use_fast: bool = False) -> AutoTokenizer:
    """
    トークナイザーをロードする関数
    
    Args:
        model_name: 使用するトークナイザー名（デフォルトはmegagonlabs/t5-base-japanese-web）
        tokenizer_path: 既存のトークナイザーファイルへのパス（指定された場合はそちらを優先）
        save_dir: トークナイザーを保存するディレクトリ（指定された場合は保存を試みる）
        use_fast: 高速トークナイザーを使用するかどうか
        
    Returns:
        設定済みのトークナイザー
    """
    import os
    import json
    from transformers import AutoTokenizer
    
    # 優先順位:
    # 1. 直接指定されたトークナイザーパス
    # 2. save_dir内のトークナイザー
    # 3. 既定の場所のトークナイザー
    # 4. Hugging Faceからダウンロード
    
    tokenizer = None
    tokenizer_loaded_from = None
    
    # 1. 直接指定されたトークナイザーパスを確認
    if tokenizer_path and os.path.exists(tokenizer_path):
        try:
            # ファイル形式を判断
            if tokenizer_path.endswith('.json'):
                # JSONファイルからトークナイザーをロード
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=use_fast)
                tokenizer_loaded_from = f"指定されたパス: {tokenizer_path}"
            elif tokenizer_path.endswith('.pkl'):
                # Pickleファイルからトークナイザーをロード
                import pickle
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
                tokenizer_loaded_from = f"指定されたPickleファイル: {tokenizer_path}"
            else:
                # ディレクトリからトークナイザーをロード
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=use_fast)
                tokenizer_loaded_from = f"指定されたディレクトリ: {tokenizer_path}"
        except Exception as e:
            print(f"指定されたトークナイザーの読み込みに失敗しました: {e}")
    
    # 2. データディレクトリにあるトークナイザーを確認
    if tokenizer is None and save_dir:
        # 標準の保存場所を確認
        tokenizer_default_dir = os.path.join(save_dir, "tokenizers")
        tokenizer_json_path = os.path.join(tokenizer_default_dir, "tokenizer_model.json")
        tokenizer_pkl_path = os.path.join(tokenizer_default_dir, "tokenizer.pkl")
        
        # JSONファイルの確認
        if os.path.exists(tokenizer_json_path):
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_default_dir, use_fast=use_fast)
                tokenizer_loaded_from = f"データディレクトリ内のJSON: {tokenizer_json_path}"
            except Exception as e:
                print(f"データディレクトリ内のJSONトークナイザーの読み込みに失敗しました: {e}")
        
        # Pickleファイルの確認
        if tokenizer is None and os.path.exists(tokenizer_pkl_path):
            try:
                import pickle
                with open(tokenizer_pkl_path, 'rb') as f:
                    tokenizer = pickle.load(f)
                tokenizer_loaded_from = f"データディレクトリ内のPickle: {tokenizer_pkl_path}"
            except Exception as e:
                print(f"データディレクトリ内のPickleトークナイザーの読み込みに失敗しました: {e}")
    
    # 3. 既定場所の確認 (常に高速ストレージが最優先)
    if tokenizer is None:
        default_tokenizer_paths = [
            # 高速ストレージを最優先
            "/content/fast_data/tokenizers/tokenizer.pkl",
            "/content/fast_data/tokenizers/tokenizer_model.json",
            # ドライブの既定パス
            "/content/drive/MyDrive/slm/data/fujiki/wiki40b_ja/tokenizers/tokenizer.pkl",
            "/content/drive/MyDrive/slm/data/fujiki/wiki40b_ja/tokenizers/tokenizer_model.json",
            "/content/drive/MyDrive/slm/checkpoints/tokenizers/tokenizer_model.json",
        ]
        
        for path in default_tokenizer_paths:
            if os.path.exists(path):
                try:
                    # Pickleファイルの場合
                    if path.endswith('.pkl'):
                        import pickle
                        with open(path, 'rb') as f:
                            tokenizer = pickle.load(f)
                        tokenizer_loaded_from = f"既定のPickleパス: {path}"
                    else:
                        # JSON/ディレクトリの場合
                        tokenizer_dir = os.path.dirname(path)
                        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=use_fast)
                        tokenizer_loaded_from = f"既定のパス: {path}"
                    break
                except Exception as e:
                    print(f"既定のパス {path} からのトークナイザー読み込みに失敗しました: {e}")
    
    # 4. Hugging Faceからダウンロード
    if tokenizer is None:
        try:
            print(f"Hugging Faceから '{model_name}' トークナイザーをダウンロード中...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
            tokenizer_loaded_from = f"Hugging Face: {model_name}"
            
            # トークナイザーを保存（指定されたディレクトリがある場合）
            if save_dir:
                tokenizer_save_dir = os.path.join(save_dir, "tokenizers")
                os.makedirs(tokenizer_save_dir, exist_ok=True)
                
                # トークナイザーをJSON形式で保存
                try:
                    tokenizer.save_pretrained(tokenizer_save_dir)
                    print(f"トークナイザーをディレクトリに保存しました: {tokenizer_save_dir}")
                except Exception as e:
                    print(f"トークナイザーの保存に失敗しました: {e}")
                
                # トークナイザーをPickle形式でも保存
                try:
                    import pickle
                    tokenizer_pkl_path = os.path.join(tokenizer_save_dir, "tokenizer.pkl")
                    with open(tokenizer_pkl_path, 'wb') as f:
                        pickle.dump(tokenizer, f)
                    print(f"トークナイザーをPickleファイルとして保存しました: {tokenizer_pkl_path}")
                except Exception as e:
                    print(f"トークナイザーのPickle保存に失敗しました: {e}")
                
                # 高速アクセス用には直接データセットと一緒にコピーされるためここでは追加コピーしない
                # (データセットコピー時にディレクトリごとコピーされる)
        except Exception as e:
            print(f"Hugging Faceからのトークナイザーダウンロードに失敗しました: {e}")
            raise ValueError(f"トークナイザーを読み込めませんでした: {e}")
    
    # マスクトークンがない場合は追加
    if not hasattr(tokenizer, 'mask_token') or tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '<mask>'})
        print(f"マスクトークン '<mask>' を追加しました。マスクトークンID: {tokenizer.mask_token_id}")
    
    # BOSトークンがない場合は追加 
    if not hasattr(tokenizer, 'bos_token') or tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '<s>'})
        print(f"BOSトークン '<s>' を追加しました。")
    
    # トークナイザー情報を表示
    print(f"トークナイザー情報:")
    print(f"  ロード元: {tokenizer_loaded_from}")
    print(f"  クラス: {type(tokenizer).__name__}")
    print(f"  語彙サイズ: {tokenizer.vocab_size}")
    print(f"  マスクトークン: {tokenizer.mask_token}, ID: {tokenizer.mask_token_id}")
    
    return tokenizer

def tokenize_batch(tokenizer: AutoTokenizer, texts: List[str], max_length: int = 512) -> Dict[str, List[List[int]]]:
    """
    テキストのバッチをトークン化する簡易関数
    
    Args:
        tokenizer: トークナイザー
        texts: トークン化するテキストのリスト
        max_length: 最大シーケンス長
        
    Returns:
        トークン化されたバッチ（input_ids, attention_maskを含む辞書）
    """
    tokenized = {"input_ids": [], "attention_mask": []}
    
    for text in texts:
        # トークン化
        token_ids = tokenizer.encode(
            text, 
            add_special_tokens=False,
            max_length=max_length,
            truncation=True
        )
        
        # 注意マスクを作成（すべて1）
        attn_mask = [1] * len(token_ids)
        
        tokenized["input_ids"].append(token_ids)
        tokenized["attention_mask"].append(attn_mask)
    
    return tokenized