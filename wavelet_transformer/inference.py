"""
Wavelet Transformerの推論ユーティリティ
事前学習済みモデルの読み込みと推論処理
"""
import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm

from wavelet_transformer.models.wavelet_model import WaveletModelForMaskedLM
from wavelet_transformer.models.transformer_model import TransformerForMaskedLM
from wavelet_transformer.config import get_wavelet_config, get_transformer_config
from transformers import PreTrainedTokenizer, AutoTokenizer

class InferenceEngine:
    """
    WaveletTransformerとベースラインTransformerの推論エンジン
    
    MLMタスクと系列処理の推論、及び性能比較を行います
    """
    def __init__(
        self,
        model_path: str,
        tokenizer_path_or_name: str,
        model_type: str = "wavelet", # "wavelet" または "transformer"
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model_path: モデルチェックポイントのパス
            tokenizer_path_or_name: トークナイザのパス or Hugging Face ID
            model_type: モデルの種類
            device: 推論デバイス (デフォルト: 自動検出)
        """
        self.model_type = model_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # トークナイザーのロード
        self.tokenizer = self._load_tokenizer(tokenizer_path_or_name)
        
        # チェックポイントからモデル設定を取得
        config = self._get_config_from_checkpoint(model_path)
        
        # モデルの初期化
        self.model = self._initialize_model(config)
        
        # チェックポイントをロード
        self._load_checkpoint(model_path)
        
        # モデルを評価モードに設定
        self.model.eval()
        
        # 性能測定用の統計
        self.inference_stats = {}
        
    def _load_tokenizer(self, tokenizer_path_or_name: str) -> PreTrainedTokenizer:
        """トークナイザーをロード"""
        try:
            if os.path.exists(tokenizer_path_or_name):
                return AutoTokenizer.from_pretrained(tokenizer_path_or_name)
            else:
                return AutoTokenizer.from_pretrained(tokenizer_path_or_name)
        except Exception as e:
            raise ValueError(f"トークナイザーのロードに失敗しました: {e}")
    
    def _get_config_from_checkpoint(self, checkpoint_path: str) -> Any:
        """チェックポイントから設定を取得"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        if "config" in checkpoint:
            return checkpoint["config"]
        elif "model_config" in checkpoint:
            return checkpoint["model_config"]
        else:
            # 設定がない場合はデフォルト設定を使用
            if self.model_type == "wavelet":
                return get_wavelet_config("base")
            else:
                return get_transformer_config("base")
    
    def _initialize_model(self, config: Any) -> torch.nn.Module:
        """設定に基づいてモデルを初期化"""
        if self.model_type == "wavelet":
            model = WaveletModelForMaskedLM(config)
        else:
            model = TransformerForMaskedLM(config)
        
        return model.to(self.device)
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """チェックポイントをロード"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # 互換性のためのフォールバック
            try:
                self.model.load_state_dict(checkpoint)
            except Exception as e:
                raise ValueError(f"モデルの状態辞書のロードに失敗しました: {e}")
                
        print(f"チェックポイントをロードしました: {checkpoint_path}")
    
    def fill_mask(
        self, 
        text: str, 
        top_k: int = 5,
        return_score: bool = True
    ) -> List[Dict[str, Any]]:
        """
        マスクされたテキストの穴埋め
        
        Args:
            text: マスクを含むテキスト（[MASK]または<mask>を含む）
            top_k: 返す候補数
            return_score: スコアも返すかどうか
            
        Returns:
            予測結果のリスト
        """
        # マスクトークンの正規化
        mask_token = self.tokenizer.mask_token
        text = text.replace("[MASK]", mask_token).replace("<mask>", mask_token)
        
        # テキストのエンコード
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # マスクトークンの位置を特定
        mask_positions = torch.where(input_ids == self.tokenizer.mask_token_id)[1]
        
        # 推論の実行
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
        # 予測結果を取得
        logits = outputs["logits"]
        
        # 各マスク位置に対する結果を収集
        results = []
        
        for mask_idx in mask_positions:
            mask_logits = logits[0, mask_idx]
            
            # 上位k個の予測を取得
            top_scores, top_indices = torch.topk(mask_logits, k=top_k)
            top_tokens = self.tokenizer.convert_ids_to_tokens(top_indices.tolist())
            
            # 結果を整形
            mask_result = []
            for i, (token, idx, score) in enumerate(zip(top_tokens, top_indices.tolist(), top_scores.tolist())):
                item = {
                    "token": token,
                    "token_id": idx,
                    "sequence": text.replace(mask_token, token, 1)
                }
                if return_score:
                    item["score"] = float(score)
                mask_result.append(item)
            
            results.append(mask_result)
        
        return results
    
    def batch_process(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: Optional[int] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        バッチでテキストを処理
        
        Args:
            texts: テキストのリスト
            batch_size: バッチサイズ
            max_length: 最大シーケンス長
            return_hidden_states: 隠れ状態も返すか
            
        Returns:
            モデル出力の辞書
        """
        # テキストのトークン化と入力テンソルの準備
        encoded_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # データ量に基づいてバッチを準備
        all_results = {
            "logits": [],
            "hidden_states": [] if return_hidden_states else None
        }
        
        # バッチ処理
        for i in tqdm(range(0, len(texts), batch_size), desc="バッチ処理中"):
            batch_inputs = {
                k: v[i:i+batch_size].to(self.device)
                for k, v in encoded_inputs.items()
            }
            
            # 推論実行
            with torch.no_grad():
                outputs = self.model(**batch_inputs)
            
            # 結果を収集
            all_results["logits"].append(outputs["logits"].cpu())
            
            if return_hidden_states:
                all_results["hidden_states"].append(outputs["hidden_states"].cpu())
        
        # 結果を結合
        all_results["logits"] = torch.cat(all_results["logits"], dim=0)
        
        if return_hidden_states:
            all_results["hidden_states"] = torch.cat(all_results["hidden_states"], dim=0)
        
        return all_results
    
    def measure_performance(
        self,
        sample_text: str,
        num_runs: int = 10,
        warmup_runs: int = 2
    ) -> Dict[str, float]:
        """
        推論性能を測定
        
        Args:
            sample_text: サンプルテキスト
            num_runs: 測定回数
            warmup_runs: ウォームアップ実行数
            
        Returns:
            性能メトリクスの辞書
        """
        # 入力の準備
        inputs = self.tokenizer(sample_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # ウォームアップ実行
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
        
        # 時間測定のためのCUDA同期
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # 測定実行
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        execution_times = []
        
        for _ in range(num_runs):
            if self.device.type == "cuda":
                start_time.record()
            else:
                start = torch.utils.time.time()
                
            # 推論実行
            with torch.no_grad():
                _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            if self.device.type == "cuda":
                end_time.record()
                torch.cuda.synchronize()
                execution_times.append(start_time.elapsed_time(end_time) / 1000)  # ミリ秒→秒
            else:
                execution_times.append(torch.utils.time.time() - start)
        
        # 統計計算
        avg_time = sum(execution_times) / len(execution_times)
        seq_length = input_ids.shape[1]
        tokens_per_second = seq_length / avg_time
        
        # メモリ使用量（GPUの場合）
        if self.device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
        else:
            memory_allocated = 0
            memory_reserved = 0
            
        # 結果を収集
        stats = {
            "avg_time_seconds": avg_time,
            "tokens_per_second": tokens_per_second,
            "sequence_length": seq_length,
            "memory_allocated_mb": memory_allocated,
            "memory_reserved_mb": memory_reserved,
            "model_type": self.model_type
        }
        
        self.inference_stats = stats
        return stats
    
    def get_wavelet_representations(
        self, 
        text: str,
        layer_index: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        ウェーブレット層の内部表現を取得
        
        Args:
            text: 入力テキスト
            layer_index: 取得する層のインデックス
            
        Returns:
            ウェーブレット表現の辞書
        """
        # このメソッドはwaveletモデルでのみ使用可能
        if self.model_type != "wavelet":
            raise ValueError("このメソッドはウェーブレットモデルでのみ利用できます")
        
        # 入力の準備
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # 波動表現をキャプチャするためのフック
        wavelet_outputs = {}
        
        def hook_fn(module, input, output):
            wavelet_outputs["wavelet_output"] = output.detach().cpu()
            
        # ウェーブレット層にフックを登録
        target_layer = self.model.wavelet_model.encoder_layers[layer_index].wavelet
        handle = target_layer.register_forward_hook(hook_fn)
        
        # 推論実行
        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # フックを除去
        handle.remove()
        
        # トークンを取得
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        
        return {
            "wavelet_output": wavelet_outputs["wavelet_output"],
            "tokens": tokens
        }
