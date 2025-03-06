# slm/evaluation.py

"""
How:
    言語モデルの評価基準を複数導入します。
    - perplexity (PPL)
    - BLEU
    - ROUGE
    などを計算できるようにし、
    デコードには temperature を用いた確率サンプリングを実装。

Why not:
    Greedy だと多様性が乏しいため、temperature で出力分布を制御し、多様性を獲得。

What:
    evaluation.py から他のファイルで呼び出されることで、複数評価指標を統一的に計算可能。
"""

from typing import Optional
import torch
from torch.utils.data import DataLoader, Dataset
from slm.cce_loss import CceLoss
from slm.modules.wave_network import WaveNetworkLM

import evaluate

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")


def evaluate_perplexity(
    model: WaveNetworkLM,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 8
) -> float:
    """
    How:
        クロスエントロピーからPerplexityを計算する。
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = CceLoss(ignore_index=-100)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            loss_val = criterion(logits, labels) * labels.numel()
            total_loss += loss_val.item()
            total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss))
    return ppl.item()


def evaluate_bleu(
    model: WaveNetworkLM,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 8,
    max_new_tokens: int = 32,
    temperature: float = 1.0
) -> float:
    """
    How:
        BLEUスコアを計測するための関数。
        dataset の各サンプルに 'target_text' があると想定し、
        temperature を使った確率サンプリングでモデル生成を行い比較。

    Why not:
        Greedy 生成では多様性が得られず BLEUスコアが一面的な評価に留まるため、
        temperature で分布を調整して評価する。

    What:
        - ここでは最小限の実装で、分かち書きや Tokenizer.decode() を簡易化している。
        - 本番環境では形態素解析や正規化処理が必要。
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_references = []
    all_hypotheses = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            # 参照文
            references = [[ref_str.split()] for ref_str in batch["target_text"]]

            outputs = []
            for i in range(input_ids.size(0)):
                single_input = input_ids[i].unsqueeze(0)
                generated = temperature_sampling_decode(
                    model,
                    single_input,
                    max_new_tokens,
                    device,
                    temperature=temperature
                )
                outputs.append(generated)

            # BLEU用に単語列にsplit
            hypotheses = [out_str.split() for out_str in outputs]

            all_references.extend(references)
            all_hypotheses.extend(hypotheses)

    results = bleu_metric.compute(
        predictions=all_hypotheses,
        references=all_references
    )
    return results["bleu"]


def evaluate_rouge(
    model: WaveNetworkLM,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 8,
    max_new_tokens: int = 32,
    temperature: float = 1.0
) -> dict:
    """
    How:
        ROUGEスコアを測るための関数。
        dataset の各サンプルに 'target_text' があると想定し、
        temperature サンプリングで生成したテキストを比較。

    Why not:
        Greedy だと確率最大トークン一辺倒になりがちなので、多様性を出すため temperature を活用。

    What:
        - ここでも単純なsplitを用いており、本格利用には正規化などが必要。
        - 戻り値は ROUGE の各種指標をまとめた辞書。
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    references = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            ref_texts = batch["target_text"]

            gen_texts = []
            for i in range(input_ids.size(0)):
                single_input = input_ids[i].unsqueeze(0)
                generated = temperature_sampling_decode(
                    model,
                    single_input,
                    max_new_tokens,
                    device,
                    temperature=temperature
                )
                gen_texts.append(generated)

            predictions.extend(gen_texts)
            references.extend(ref_texts)

    results = rouge_metric.compute(
        predictions=predictions,
        references=references
    )
    return results


def temperature_sampling_decode(
    model: WaveNetworkLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 1.0
) -> str:
    """
    How:
        temperature に基づいて、モデル出力ロジットをソフトマックス→1トークンをランダムサンプリングし、
        逐次生成を行う簡易関数。

    Why not:
        Greedy だと多様性が失われるため、temperature を使って確率分布を変形し、
        多様性を制御する。

    What:
        - 実際には Tokenizer が必要。ここでは ID列を str() で連結しているだけ。
        - top-k や top-p と組み合わせるとさらに良い。
    """
    model.eval()
    generated = input_ids.clone().to(device)

    for _ in range(max_new_tokens):
        logits = model(generated)  # (1, seq_len, vocab_size)
        next_token_logits = logits[:, -1, :]  # (1, vocab_size)

        # temperature スケーリング
        next_token_logits = next_token_logits / temperature

        probs = torch.softmax(next_token_logits, dim=-1)  # (1, vocab_size)
        # サンプリング
        next_token_id = torch.multinomial(probs, num_samples=1)  # (1,1)

        generated = torch.cat([generated, next_token_id], dim=1)

    # ID列をそのまま文字列に (本来はTokenizer.decode)
    tokens = generated.squeeze(0).cpu().tolist()
    out_str = " ".join(str(tk) for tk in tokens)
    return out_str


"""
このサンプルでは batch["target_text"] が存在する前提にしています。

QAタスク の場合は answers["text"] を参照としたり、翻訳 の場合は en_text → ja_text のようなペアを持ちます。
分かち書き や 句読点の扱い は評価指標によって調整してください。
greedy_decode() は非常に単純な実装です。本格的にやるなら transformers の generate() API (Beam Search, Top-k サンプリングなど) を利用し、出力を Tokenizer で decode する必要があります。
"""