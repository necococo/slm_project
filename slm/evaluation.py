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
from cut_cross_entropy import linear_cross_entropy
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
        WaveNetworkLMモデルのCut Cross Entropy対応。
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # 埋め込みを取得
            embeddings = model(input_ids)
            # 分類器の重みを取得
            classifier = model.get_classifier_weights()
            
            # 重要: linear_cross_entropyに渡す前に精度変換
            # デバイスに応じた精度変換を行う
            if device.type == "cuda":
                # GPUの場合はfloat16に変換
                embeddings = embeddings.half()
                classifier = classifier.half()
            elif hasattr(torch, 'bfloat16') and (device.type == "xla" or device.type == "cpu"):
                # TPUやCPUでbfloat16が利用可能な場合
                embeddings = embeddings.to(torch.bfloat16)
                classifier = classifier.to(torch.bfloat16)
            
            # linear_cross_entropyで損失を計算
            loss = linear_cross_entropy(embeddings, classifier, labels)
            
            # 有効なトークン数を計算（-100はignore_index）
            valid_tokens = (labels != -100).sum().item()
            
            # 損失と有効トークン数を累積
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    # 平均損失を計算
    avg_loss = total_loss / max(total_tokens, 1)  # ゼロ除算を防ぐ
    
    # パープレキシティの計算
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
                
                # データセットからトークナイザーを取得（可能であれば）
                tokenizer = None
                if hasattr(model, 'config') and hasattr(model.config, 'tokenizer'):
                    tokenizer = model.config.tokenizer
                
                generated = temperature_sampling_decode(
                    model,
                    single_input,
                    max_new_tokens,
                    device,
                    temperature=temperature,
                    tokenizer=tokenizer  # トークナイザーを渡す
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
                
                # データセットからトークナイザーを取得（可能であれば）
                tokenizer = None
                if hasattr(model, 'config') and hasattr(model.config, 'tokenizer'):
                    tokenizer = model.config.tokenizer
                
                generated = temperature_sampling_decode(
                    model,
                    single_input,
                    max_new_tokens,
                    device,
                    temperature=temperature,
                    tokenizer=tokenizer  # トークナイザーを渡す
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
    temperature: float = 1.0,
    tokenizer: Optional[object] = None
) -> str:
    """
    How:
        temperature に基づいて、モデル出力から次トークンをランダムサンプリングし、
        逐次生成を行う関数。トークナイザーを使用して自然なテキスト生成を行う。

    Args:
        model: WaveNetworkLMモデル
        input_ids: 入力トークンID [batch_size=1, seq_len]
        max_new_tokens: 生成する最大トークン数
        device: 使用するデバイス
        temperature: サンプリング温度 (1.0で等確率、小さいほどgreedy)
        tokenizer: テキストデコード用のトークナイザー（オプション）
        
    Returns:
        生成されたテキスト
    """
    model.eval()
    generated = input_ids.clone().to(device)
    original_len = len(input_ids[0])

    for _ in range(max_new_tokens):
        # モデルから埋め込みを取得
        embeddings = model(generated)  # (1, seq_len, hidden_size)
        # 分類器の重みを取得
        classifier = model.get_classifier_weights()  # (vocab_size, hidden_size)
        
        # 重要: デバイス種類に応じた精度変換
        if device.type == "cuda":
            # GPUの場合はfloat16に変換
            embeddings = embeddings.half()
            classifier = classifier.half()
        elif hasattr(torch, 'bfloat16') and (device.type == "xla" or device.type == "cpu"):
            # TPUやCPUでbfloat16が利用可能な場合
            embeddings = embeddings.to(torch.bfloat16)
            classifier = classifier.to(torch.bfloat16)
        
        # 最後のトークンの埋め込みのみを使用
        last_embedding = embeddings[:, -1, :]  # (1, hidden_size)
        
        # 内積計算でスコア行列を取得
        scores = torch.matmul(last_embedding, classifier.T)  # (1, vocab_size)

        # temperature スケーリング
        scores = scores / max(temperature, 1e-6)  # ゼロ除算を防止

        # 確率分布に変換
        probs = torch.softmax(scores, dim=-1)  # (1, vocab_size)
        
        # サンプリング
        next_token_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # 生成したトークンを追加
        generated = torch.cat([generated, next_token_id], dim=1)

    # 生成されたトークンIDを取得（入力部分を除く）
    generated_ids = generated[0, original_len:].cpu().tolist()
    
    # トークナイザーが提供されていれば、デコードに使用
    if tokenizer is not None:
        try:
            # トークナイザーのdecodeメソッドを使用
            if hasattr(tokenizer, 'decode') and callable(tokenizer.decode):
                return tokenizer.decode(generated_ids, skip_special_tokens=True)
        except Exception as e:
            print(f"トークナイザーによるデコードに失敗しました: {e}")
    
    # トークナイザーがない場合や失敗時はIDを文字列として返す（フォールバック）
    return " ".join(str(token_id) for token_id in generated_ids)


"""
このサンプルでは batch["target_text"] が存在する前提にしています。

QAタスク の場合は answers["text"] を参照としたり、翻訳 の場合は en_text → ja_text のようなペアを持ちます。
分かち書き や 句読点の扱い は評価指標によって調整してください。
greedy_decode() は非常に単純な実装です。本格的にやるなら transformers の generate() API (Beam Search, Top-k サンプリングなど) を利用し、出力を Tokenizer で decode する必要があります。
"""