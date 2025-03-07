# Train Small, Infer Large: Memory-Efficient LoRA Training for Large Language Models

## 1. 背景
- **Large Language Models (LLMs)**  
  - 例: GPT-4, LLaMA, PaLMなど。
  - 特徴: 400億パラメータを超えるモデルもあり、タスク汎化性が優れている。
- **Fine-tuningの課題**  
  - 目的: タスク固有のカスタマイズに重要。
  - 問題: メモリ要求が膨大。
    - 例: LLaMA-70Bのfine-tuningには16-bit精度で1178GBのメモリが必要（15台のA100-80G GPU）。
- **Low-Rank Adaptation (LoRA)**  
  - 概要: 効率的なfine-tuning手法。
  - 仕組み: 元のモデルパラメータをfreezeし、低ランクのadapter matricesをトレーニング。
    - 利点: トレーニングパラメータ数を大幅削減（例: LLaMA-2-13Bで406倍削減）。
    - 課題: トレーニング時、元のモデルパラメータのメモリ使用量が支配的。
      - 例: LLaMA-2-13Bでは元のモデルが26GB（FP16）、LoRA adapterが64MB（BF16）だが、元のモデルをメモリに保持する必要あり。

## 2. 問題提起
- **LoRAの限界**  
  - トレーニング時のメモリ使用量が元のモデルパラメータに依存。
  - 例: LLaMA-2-13Bで量子化（4-bit）しても6.5GB必要。
- **研究質問**  
  - 「inferenceの精度を維持しつつ、LoRAトレーニングのメモリオーバーヘッドをさらに削減できるか？」

## 3. 提案手法: LoRAM
- **概要**  
  - 名称: Memory-efficient LoRA training (LoRAM)。
  - アイデア: トレーニング時にはpruned (small) modelを使い、inference時にはoriginal (large) modelに回復した低ランク行列を使用。
- **直感**  
  - LLMsの多くのニューロンはトレーニング時の有用性が低い（冗長）が、inference時には重要。
- **プロセス**  
  1. **Pruned Full-Rank Weight Generation**  
     - 手法: pruning algorithm（例: LLM-Pruner, SparseGPT）でoriginal modelからpruned modelを生成。
     - 例: 構造化（structured）または非構造化（unstructured） pruning。
  2. **Pruned Low-Rank Matrix Training**  
     - 手法: pruned model上でLoRAトレーニングを行い、pruned low-rank matrices（Bᴾ, Aᴾ）を学習。
     - 計算: h = xW₀ᴾ + x(BᴾAᴾ)。
  3. **Recovered Low-Rank Matrix Generation**  
     - 手法: 学習したpruned low-rank matricesをoriginal modelの次元に回復（Bᴿ*, Aᴿ*）。
     - 処理: 剪定された位置にゼロを埋め、元の形状に適合。
  4. **Recovered Low-Rank Matrix Inference**  
     - 手法: inference時にoriginal modelに回復した低ランク行列を適用。
     - 計算: h = x(W₀ + Bᴿ*Aᴿ*)。
- **課題と解決策**  
  - **知識の不一致**: pruningによりpruned modelとoriginal modelの知識に差が生じる。
  - **Alignment Strategy**: pruned modelを小規模データセット（約1億500万トークン）でcontinual pre-training。
    - 効果: 知識の不一致を低コストで解消。
    - 実装: モデル公開者がオフラインで一度実行可能（例: Meta AIがLLaMA用に提供）。
- **拡張: QLoRAM**  
  - 手法: LoRAMに量子化（例: 4-bit QLoRA）を組み合わせ、さらにメモリ削減。

## 4. 技術的詳細
- **LoRAのおさらい**  
  - 式: W = W₀ + WΔ = W₀ + BA（W₀はfreeze、BとAを更新）。
  - メモリ: W₀が支配的（例: LLaMA-2-13Bで6.5GB vs. WΔで576MBピーク）。
- **LoRAMの改良**  
  - トレーニング: pruned model（W₀ᴾ）で低ランク行列を学習。
  - 回復: W₀に適用可能なWΔᴿ*を生成。
  - 量子化: W₀ᴾを4-bitに圧縮（QLoRAM）。
- **実装のポイント**  
  - 構造化pruning: 物理的に重みを削除し、コンパクトなdense行列に。
  - 非構造化pruning: 次元は変わらず、スパース行列としてゼロを保持。

## 5. 実験結果
- **対象モデル**  
  - LLaMA-2-13B, LLaMA-2-70B, LLaMA-3.1-70B。
- **Fine-tuning Convergence**  
  - データセット: OpenHermes, OpenOrca。
  - 評価: out-of-domain（Alpaca）とin-domainのperplexity。
  - 結果: LoRAMはLoRAと同等の収束性を示しつつ、メモリ使用量を削減。
- **Downstream Task Performance**  
  - タスク: 数学的推論（MathQA, GSM8K）、常識的推論（CSR）、コード生成（HumanEval）。
  - 例: LLaMA-2-70BでQLoRAM使用時、パラメータ12.84倍削減しつつGSM8Kで52%→57%に向上。
- **メモリ削減効果**  
  - LLaMA-2-70B: 16.95倍削減（QLoRAM）。
  - LLaMA-3.1-70B: 15.81倍削減。
  - 実用例: 70Bモデルのトレーニングが20GB HBMのGPUで可能（従来はA100-80G必要）。
- **Ablation Study**  
  - Recoveryの必要性: 回復なしではperplexityが悪化。
  - Alignmentの効果: 小規模データでの事前学習で性能向上。
- **Scaling Laws**  
  - パラメータ削減率を増やしても性能維持（最大28.56倍まで検証）。

## 6. 利点と意義
- **主要利点**  
  - トレーニング時のメモリ使用量を大幅削減。
  - inference時にoriginal modelの性能を活用。
  - 既存の量子化手法（QLoRA）と互換性あり。
- **エンジニア視点での価値**  
  - ハードウェア制約の緩和: 消費者向けGPU（20GB）で70Bモデルをfine-tuning可能。
  - 効率性: 計算コストとストレージコストのバランスが良い。
  - 応用性: LLaMAシリーズ以外への拡張も期待。

## 7. 結論と今後の展望
- **結論**  
  - LoRAMはメモリ効率の良いLoRAトレーニングを実現。
  - 例: LLaMA-2-70Bで16.95倍のパラメータ削減、性能向上を両立。
- **今後の研究**  
  - コンテキスト対応の計算グラフ回復。
  - 他のモデル（例: Vision Transformers, Diffusion Models）への適用。
- **コミュニティへの貢献**  
  - ソースコード公開: [GitHub](https://github.com/junzhang-zj/LoRAM)。
  - 大規模モデルのfine-tuningを民主化。