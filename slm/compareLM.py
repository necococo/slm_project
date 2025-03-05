# slm/compareLM.py

"""
How:
    slm のモデル (ours) と Hugging Face のモデル (hf_model_name) を
    同じデータセットに対して評価し、比較する。

    ここでも temperature サンプリングを使って出力を生成・評価する方針に修正。

Why not:
    Greedy だと単調な出力になりやすい。多様性を評価するため temperature を考慮。

What:
    - PPL はモデル内部のloss計算なので特に temperature は関係ない
    - BLEU や ROUGE など生成タスクでは temperature で出力多様性を制御する
    - CSVに結果を保存する
"""

import os
import csv
import torch
from typing import Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

from slm.model import WaveHierarchicalLM
from slm.evaluation import (
    evaluate_perplexity,
    evaluate_bleu,
    evaluate_rouge
)

def compare_two_models(
    ours: WaveHierarchicalLM,
    hf_model_name: str,
    dataset,
    device: torch.device,
    output_csv: str,
    temperature: float = 1.0
) -> None:
    """
    How:
        ours モデルと huggingface の hf_model_name をロードしたモデルを
        同一データで評価する。すべて temperature サンプリングで生成を行い、
        BLEUやROUGEを計測する。

    Why not:
        Greedy 生成を廃止し、temperature で多様性を制御する形に統一。

    What:
        - PPL は学習ログから計算されるため temperature は影響しない
        - BLEU, ROUGE は 生成出力に依存するので temperature を調整可能
    """
    # slm モデルの指標 (PPLはtemperature無関係)
    ours_ppl = evaluate_perplexity(ours, dataset, device)
    ours_bleu = evaluate_bleu(ours, dataset, device, temperature=temperature)
    ours_rouge = evaluate_rouge(ours, dataset, device, temperature=temperature)

    # Hugging Face model
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    hf_model.to(device)

    hf_ppl = evaluate_perplexity_hf(hf_model, hf_tokenizer, dataset, device)
    hf_bleu = evaluate_bleu_hf(hf_model, hf_tokenizer, dataset, device, temperature=temperature)
    hf_rouge = evaluate_rouge_hf(hf_model, hf_tokenizer, dataset, device, temperature=temperature)

    # 表示
    print("=== Our Model ===")
    print(f"PPL:   {ours_ppl:.4f}")
    print(f"BLEU:  {ours_bleu:.4f}")
    print("ROUGE:", ours_rouge)

    print("=== HF Model: ", hf_model_name, "===")
    print(f"PPL:   {hf_ppl:.4f}")
    print(f"BLEU:  {hf_bleu:.4f}")
    print("ROUGE:", hf_rouge)

    # CSVに保存
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode="w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "PPL", "BLEU", "ROUGE1", "ROUGE2", "ROUGEL"])
        writer.writerow([
            "WaveHierarchicalLM",
            f"{ours_ppl:.4f}",
            f"{ours_bleu:.4f}",
            f"{ours_rouge['rouge1']:.4f}",
            f"{ours_rouge['rouge2']:.4f}",
            f"{ours_rouge['rougeL']:.4f}"
        ])
        writer.writerow([
            hf_model_name,
            f"{hf_ppl:.4f}",
            f"{hf_bleu:.4f}",
            f"{hf_rouge['rouge1']:.4f}",
            f"{hf_rouge['rouge2']:.4f}",
            f"{hf_rouge['rougeL']:.4f}"
        ])
    print("結果をCSVに保存しました:", output_csv)


def evaluate_perplexity_hf(
    hf_model: AutoModelForCausalLM,
    hf_tokenizer: AutoTokenizer,
    dataset,
    device: torch.device,
    batch_size: int = 8
) -> float:
    """
    How:
        HFモデルに対して Perplexity を計測。
        temperature は PPL計算に影響しないため不要。

    Why not:
        PPLはモデル内部の確率(対数尤度)で計算されるため、生成戦略と独立。

    What:
        waveネットワークモデルとの計測手法を揃えるため、CrossEntropyLossベースで計算。
    """
    hf_model.eval()
    from torch.utils.data import DataLoader
    from slm.cce_loss import CceLoss
    import math

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = CceLoss(ignore_index=-100)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = hf_model(input_ids, labels=labels)
            loss_val = outputs.loss * labels.numel()
            total_loss += loss_val.item()
            total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss))
    return ppl.item()


def evaluate_bleu_hf(
    hf_model: AutoModelForCausalLM,
    hf_tokenizer: AutoTokenizer,
    dataset,
    device: torch.device,
    batch_size: int = 8,
    max_new_tokens: int = 32,
    temperature: float = 1.0
) -> float:
    """
    How:
        HFモデルで BLEU を測定。temperature を使ってサンプリングし、出力の多様性を制御。

    Why not:
        Greedy では単調になるので、temperature によるサンプリングで生成テキストを評価。

    What:
        - top-k, top-p と組み合わせればさらに多様性調整が可能。
        - 'target_text' がバッチに含まれている前提。
    """
    import evaluate
    from torch.utils.data import DataLoader

    bleu = evaluate.load("bleu")
    hf_model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_references = []
    all_hypotheses = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids_list = batch["input_ids"]
            ref_texts = batch["target_text"]

            references = [[ref.split()] for ref in ref_texts]
            predictions = []
            for i, input_ids_raw in enumerate(input_ids_list):
                input_ids = input_ids_raw.unsqueeze(0).to(device)
                # temperature サンプリング
                gen_tokens = hf_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,           # サンプリングを有効化
                    temperature=temperature,  # ここで temperature を指定
                    num_beams=1
                )
                gen_str = hf_tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                predictions.append(gen_str)

            hypotheses = [p.split() for p in predictions]
            all_references.extend(references)
            all_hypotheses.extend(hypotheses)

    result = bleu.compute(
        predictions=all_hypotheses,
        references=all_references
    )
    return result["bleu"]


def evaluate_rouge_hf(
    hf_model: AutoModelForCausalLM,
    hf_tokenizer: AutoTokenizer,
    dataset,
    device: torch.device,
    batch_size: int = 8,
    max_new_tokens: int = 32,
    temperature: float = 1.0
) -> dict:
    """
    How:
        HFモデルで ROUGE を測定。temperature サンプリングで生成し評価。

    Why not:
        Greedy でなく分布サンプリングするので、多少変動が出るが多様性評価が可能。

    What:
        - ここでは簡易実装。top-k 等と組み合わせるのも一案。
        - 'target_text' を参照として想定。
    """
    import evaluate
    from torch.utils.data import DataLoader

    rouge = evaluate.load("rouge")
    hf_model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids_list = batch["input_ids"]
            ref_texts = batch["target_text"]

            predictions = []
            for i, input_ids_raw in enumerate(input_ids_list):
                input_ids = input_ids_raw.unsqueeze(0).to(device)
                gen_tokens = hf_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    num_beams=1
                )
                gen_str = hf_tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                predictions.append(gen_str)

            all_predictions.extend(predictions)
            all_references.extend(ref_texts)

    return rouge.compute(
        predictions=all_predictions,
        references=all_references
    )