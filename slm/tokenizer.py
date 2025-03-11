# slm/tokenizer.py
# Why not: NovelAI/nerdstash-tokenizer-v2の組み込みも想定して構成をシンプルにしてある。

import os
import shutil
from typing import List, Optional
import sentencepiece as spm
from huggingface_hub import hf_hub_download

class JapaneseTokenizer:
    """
    How:
        日本語テキストをトークン化するクラスです。
        SentencePieceモデルファイルを指定して利用可能です、NovelAI/nerdstash-tokenizer-v2 なども利用できます。
    """

    def __init__(self, model_file: Optional[str] = None, hf_model: Optional[str] = None, 
                 save_to: Optional[str] = None, filename: str = "tokenizer.model") -> None:
        """
        Args:
            model_file: ローカルのSentencePieceモデルファイルパス
            hf_model: Hugging Faceのモデル名（例："cl-tohoku/bert-base-japanese-whole-word-masking"）
            save_to: ダウンロードしたモデルを保存するパス
            filename: Hugging Faceからダウンロードするファイル名
        """
        if hf_model:
            # HuggingFaceからモデルをダウンロード
            model_path = hf_hub_download(repo_id=hf_model, filename=filename)
            
            # 指定された場所に保存
            if save_to:
                os.makedirs(os.path.dirname(save_to), exist_ok=True)
                shutil.copy(model_path, save_to)
                print(f"モデルを保存しました: {save_to}")
                self.model_path = save_to
                self.sp = spm.SentencePieceProcessor(model_file=save_to)
            else:
                self.model_path = model_path
                self.sp = spm.SentencePieceProcessor(model_file=model_path)
        elif model_file:
            self.model_path = model_file
            self.sp = spm.SentencePieceProcessor(model_file=model_file)
        else:
            raise ValueError("model_fileまたはhf_modelを指定してください")
            
        # 特殊トークン (transformers互換性のため)
        self.mask_token = "[MASK]" 
        self.mask_token_id = self.sp.piece_to_id(self.mask_token) if self.sp.piece_to_id(self.mask_token) >= 0 else 4
        # パディングトークンとIDを追加
        self.pad_token = "<pad>"
        self.pad_token_id = 0  # T5トークナイザーでは0がパディングトークンID

    def encode(self, text: str) -> List[int]:
        """
        How:
            テキストをID列に変換します。
        """
        return self.sp.encode(text, out_type=int)
    
    def encode_no_special(self, text: str) -> List[int]:
        """
        How:
            特殊トークン（BOS、EOS）を追加せずにテキストをID列に変換します。
            Hugging FaceのTokenizerのadd_special_tokens=Falseと同等です。
        """
        if hasattr(self, '_tokenizer'):
            # transformersトークナイザーを使用
            return self._tokenizer.encode(text, add_special_tokens=False)
        else:
            # 通常のencodeを使用し、末尾のEOSトークン（通常は1）を取り除く
            tokens = self.sp.encode(text, out_type=int)
            # EOSトークンが含まれている場合、取り除く
            if tokens and tokens[-1] == 1:  # 1はよくあるEOSトークンID
                return tokens[:-1]
            return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        How:
            ID列をテキストに変換します。
            
        Args:
            token_ids: デコードするトークンIDのリスト
            skip_special_tokens: 特殊トークン（</s>など）をスキップするか
            
        Returns:
            デコードされたテキスト
        """
        if hasattr(self, '_tokenizer') and hasattr(self._tokenizer, 'decode'):
            # transformersトークナイザーの場合、skip_special_tokensオプションを使用
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            # SentencePieceの場合
            text = self.sp.decode(token_ids)
            
            # 特殊トークンの削除が要求された場合
            if skip_special_tokens:
                # </s>トークンを削除
                text = text.replace("</s>", "")
            
            return text

    def tokenize_batch(self, texts: List[str]) -> List[List[int]]:
        """
        How:
            複数のテキストをまとめてトークン化します。
        """
        return [self.encode(t) for t in texts]
    
    @property
    def get_model_path(self) -> str:
        """モデルのパスを取得します"""
        return self.model_path

    # 追加: transformersのトークナイザーを変換するためのメソッド
    @classmethod
    def from_pretrained_tokenizer(cls, tokenizer):
        """
        transformersのトークナイザーからJapaneseTokenizerを作成
        """
        instance = cls.__new__(cls)
        instance.model_path = None
        # transformersのトークナイザーオブジェクトを保存
        instance._tokenizer = tokenizer
        # sentencepieceインターフェイスをエミュレート
        class SPEmulator:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                
            def encode(self, text, out_type=int):
                # add_special_tokensをFalseにして特殊トークンを追加しない
                result = self.tokenizer.encode(text, add_special_tokens=False)
                return result
                
            def decode(self, ids, skip_special_tokens=True):
                """特殊トークンをスキップするオプション付きデコード"""
                return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
                
            def piece_to_id(self, piece):
                if piece in self.tokenizer.vocab:
                    return self.tokenizer.convert_tokens_to_ids(piece)
                return -1
                
            def get_piece_size(self):
                return len(self.tokenizer.vocab)
                
        instance.sp = SPEmulator(tokenizer)
        # 特殊トークン
        instance.mask_token = tokenizer.mask_token if hasattr(tokenizer, 'mask_token') else "[MASK]"
        instance.mask_token_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else 4
        
        # パディングトークン
        instance.pad_token = tokenizer.pad_token if hasattr(tokenizer, 'pad_token') else "<pad>"
        instance.pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
        
        return instance


