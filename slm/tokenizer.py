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
                 save_to: Optional[str] = None) -> None:
        """
        Args:
            model_file: ローカルのSentencePieceモデルファイルパス
            hf_model: Hugging Faceのモデル名（例："NovelAI/nerdstash-tokenizer-v2"）
            save_to: ダウンロードしたモデルを保存するパス
        """
        if hf_model:
            # HuggingFaceからモデルをダウンロード
            model_path = hf_hub_download(repo_id=hf_model, filename="tokenizer.model")
            
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

    def encode(self, text: str) -> List[int]:
        """
        How:
            テキストをID列に変換します。
        """
        return self.sp.encode(text, out_type=int)

    def decode(self, token_ids: List[int]) -> str:
        """
        How:
            ID列をテキストに変換します。
        """
        return self.sp.decode(token_ids)

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
    

