import os
import argparse
import logging
import time
import torch
import numpy as np
import json
from typing import Dict, List, Optional, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from wavelet_transformer.models.wavelet_transformer import WaveletTransformer
from wavelet_transformer.models.transformer import TransformerModel
from wavelet_transformer.utils.data_processing import load_data, prepare_data_for_evaluation
from wavelet_transformer.utils.evaluation import (
    calculate_perplexity, 
    calculate_accuracy, 
    run_benchmark
)
from wavelet_transformer.utils.visualization import (
    visualize_attention_patterns,
    visualize_benchmark_results,
    compare_models
)

# ロギング設定
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Wavelet TransformerとTransformerモデルの評価")
    
    # データ関連
    parser.add_argument("--eval_data", type=str, required=True, help="評価データのパス")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased", help="使用するトークナイザー")
    parser.add_argument("--max_length", type=int, default=128, help="最大シーケンス長")
    parser.add_argument("--batch_size", type=int, default=16, help="バッチサイズ")
    
    # モデル関連
    parser.add_argument("--model_type", type=str, 
                        choices=["wavelet", "transformer", "gpt-1", "all"], 
                        default="all", 
                        help="評価するモデルタイプ")
    parser.add_argument("--wavelet_model_path", type=str, help="Wavelet Transformerモデルのパス")
    parser.add_argument("--transformer_model_path", type=str, help="Transformerモデルのパス")
    parser.add_argument("--gpt1_model_path", type=str, 
                        default="openai-gpt", 
                        help="GPT-1モデルのHuggingface ID (例: 'openai-gpt')")
    
    # 評価関連
    parser.add_argument("--task", type=str, choices=["text_classification", "language_modeling"], default="language_modeling", help="評価タスク")
    parser.add_argument("--benchmark", action="store_true", help="ベンチマークを実行する")
    parser.add_argument("--benchmark_iterations", type=int, default=10, help="ベンチマーク実行回数")
    
    # 出力関連
    parser.add_argument("--output_dir", type=str, default="./results", help="結果の出力先ディレクトリ")
    parser.add_argument("--visualize", action="store_true", help="評価結果を可視化する")
    parser.add_argument("--attention_vis", action="store_true", help="アテンションパターンを可視化する")
    
    # 実行環境関連
    parser.add_argument("--device", type=str, default=None, help="使用するデバイス（例：'cuda:0', 'cpu'）")
    
    return parser.parse_args()

def load_model(model_path: str, model_type: str, device: str, task: str = "language_modeling"):
    """モデルを読み込む関数"""
    logger.info(f"{model_type}モデルを読み込み中: {model_path}")
    
    try:
        if model_type == "wavelet":
            model = WaveletTransformer.from_pretrained(model_path)
        elif model_type == "transformer":
            model = TransformerModel.from_pretrained(model_path)
        elif model_type == "gpt-1":
            # Hugging Faceからモデルを読み込む
            if task == "language_modeling":
                model = AutoModelForCausalLM.from_pretrained(model_path)
            else:  # text_classification
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        model.to(device)
        model.eval()
        return model
    
    except Exception as e:
        logger.error(f"モデルの読み込みに失敗しました: {e}")
        return None

def evaluate_model(model, data_loader, task, device, model_type=None):
    """モデルを評価する関数"""
    results = {}
    
    logger.info(f"モデル評価を開始: {task}")
    
    with torch.no_grad():
        if task == "language_modeling":
            perplexity = calculate_perplexity(model, data_loader, device, model_type)
            results["perplexity"] = perplexity
            logger.info(f"パープレキシティ: {perplexity:.4f}")
        
        if task == "text_classification":
            accuracy = calculate_accuracy(model, data_loader, device, model_type)
            results["accuracy"] = accuracy
            logger.info(f"精度: {accuracy:.4f}")
    
    return results

def main():
    """メイン関数"""
    args = parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # デバイスの設定
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用デバイス: {device}")
    
    # データの読み込み
    logger.info(f"評価データの読み込み中: {args.eval_data}")
    eval_data = load_data(args.eval_data)
    eval_texts = eval_data["texts"] if isinstance(eval_data, dict) and "texts" in eval_data else []
    
    # データの前処理
    eval_dataloader = prepare_data_for_evaluation(
        eval_data, 
        tokenizer_name=args.tokenizer, 
        task=args.task, 
        max_length=args.max_length, 
        batch_size=args.batch_size
    )
    
    # GPT-1用のデータローダーを準備（必要に応じて）
    gpt1_dataloader = None
    if args.model_type in ["gpt-1", "all"]:
        if args.task == "language_modeling":
            # GPT-1用のトークナイザーでデータローダーを準備
            gpt1_tokenizer = AutoTokenizer.from_pretrained(args.gpt1_model_path)
            gpt1_dataloader = prepare_data_for_evaluation(
                eval_data, 
                tokenizer_name=args.gpt1_model_path,
                task=args.task, 
                max_length=args.max_length, 
                batch_size=args.batch_size,
                use_custom_tokenizer=gpt1_tokenizer
            )
    
    # モデルの評価
    evaluation_results = {}
    
    # Wavelet Transformerモデルの評価
    if args.model_type in ["wavelet", "all"] and args.wavelet_model_path:
        wavelet_model = load_model(args.wavelet_model_path, "wavelet", device, args.task)
        if wavelet_model:
            logger.info("Wavelet Transformerモデルの評価を実行中...")
            wavelet_results = evaluate_model(wavelet_model, eval_dataloader, args.task, device, "wavelet")
            evaluation_results["wavelet"] = wavelet_results
    
    # Transformerモデルの評価
    if args.model_type in ["transformer", "all"] and args.transformer_model_path:
        transformer_model = load_model(args.transformer_model_path, "transformer", device, args.task)
        if transformer_model:
            logger.info("Transformerモデルの評価を実行中...")
            transformer_results = evaluate_model(transformer_model, eval_dataloader, args.task, device, "transformer")
            evaluation_results["transformer"] = transformer_results
    
    # GPT-1モデルの評価
    if args.model_type in ["gpt-1", "all"]:
        gpt1_model = load_model(args.gpt1_model_path, "gpt-1", device, args.task)
        if gpt1_model:
            logger.info("GPT-1モデルの評価を実行中...")
            dataloader_to_use = gpt1_dataloader if gpt1_dataloader else eval_dataloader
            gpt1_results = evaluate_model(gpt1_model, dataloader_to_use, args.task, device, "gpt-1")
            evaluation_results["gpt-1"] = gpt1_results
    
    # ベンチマークの実行
    benchmark_results = {}
    if args.benchmark:
        logger.info("ベンチマークテストを実行中...")
        
        if args.model_type in ["wavelet", "all"] and args.wavelet_model_path and 'wavelet_model' in locals():
            logger.info("Wavelet Transformerモデルのベンチマークを実行中...")
            wavelet_benchmark = run_benchmark(
                wavelet_model, 
                eval_dataloader, 
                iterations=args.benchmark_iterations, 
                device=device
            )
            benchmark_results["wavelet"] = wavelet_benchmark
        
        if args.model_type in ["transformer", "all"] and args.transformer_model_path and 'transformer_model' in locals():
            logger.info("Transformerモデルのベンチマークを実行中...")
            transformer_benchmark = run_benchmark(
                transformer_model, 
                eval_dataloader, 
                iterations=args.benchmark_iterations, 
                device=device
            )
            benchmark_results["transformer"] = transformer_benchmark
            
        if args.model_type in ["gpt-1", "all"] and 'gpt1_model' in locals():
            logger.info("GPT-1モデルのベンチマークを実行中...")
            dataloader_to_use = gpt1_dataloader if gpt1_dataloader else eval_dataloader
            gpt1_benchmark = run_benchmark(
                gpt1_model, 
                dataloader_to_use, 
                iterations=args.benchmark_iterations, 
                device=device,
                model_type="gpt-1"
            )
            benchmark_results["gpt-1"] = gpt1_benchmark
    
    # アテンションパターン可視化
    if args.attention_vis:
        logger.info("アテンションパターンの可視化を実行中...")
        # テストデータからサンプルテキストを選択
        sample_text = eval_texts[0] if eval_texts else "This is a sample text to visualize attention patterns."
        
        # 最大100文字に制限
        if len(sample_text) > 100:
            sample_text = sample_text[:100]
            
        visualize_attention_patterns(
            wavelet_model_path=args.wavelet_model_path if args.model_type in ["wavelet", "all"] else None,
            transformer_model_path=args.transformer_model_path if args.model_type in ["transformer", "all"] else None,
            gpt1_model_path=args.gpt1_model_path if args.model_type in ["gpt-1", "all"] else None,
            tokenizer_name=args.tokenizer,
            test_text=sample_text,
            output_path=args.output_dir,
            device=device
        )
    
    # 結果の可視化
    if args.visualize:
        logger.info("評価結果の可視化を実行中...")
        
        # 性能比較のグラフを作成
        if len(evaluation_results) > 1:
            # 精度比較
            if args.task == "text_classification":
                accuracy_data = {model_type: {"accuracy": [results["accuracy"]]} 
                                for model_type, results in evaluation_results.items() 
                                if "accuracy" in results}
                
                if accuracy_data:
                    import matplotlib
                    matplotlib.use('Agg')  # GUIなしでも動作するように
                    
                    # モデル間の精度比較グラフ
                    from wavelet_transformer.utils.visualization import compare_models
                    compare_models(
                        accuracy_data,
                        metric="accuracy",
                        title="モデル間の精度比較",
                        output_path=os.path.join(args.output_dir, "accuracy_comparison.png")
                    )
            
            # パープレキシティ比較
            if args.task == "language_modeling":
                perplexity_data = {model_type: {"perplexity": [results["perplexity"]]} 
                                  for model_type, results in evaluation_results.items()
                                  if "perplexity" in results}
                
                if perplexity_data:
                    import matplotlib
                    matplotlib.use('Agg')  # GUIなしでも動作するように
                    
                    compare_models(
                        perplexity_data,
                        metric="perplexity",
                        title="モデル間のパープレキシティ比較",
                        output_path=os.path.join(args.output_dir, "perplexity_comparison.png")
                    )
            
        # ベンチマーク結果の可視化
        if benchmark_results:
            visualize_benchmark_results(benchmark_results, args.output_dir)
            
    # 評価結果をJSONで保存
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    
    combined_results = {
        "evaluation": evaluation_results,
        "benchmark": benchmark_results,
        "args": vars(args)
    }
    
    with open(results_path, "w") as f:
        import json
        json.dump(combined_results, f, indent=2)
    
    logger.info(f"評価結果を保存しました: {results_path}")
    logger.info("評価が完了しました")

if __name__ == "__main__":
    main()
