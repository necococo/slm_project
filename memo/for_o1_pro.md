
「google colab pro+のA100 GPU一つで小さくても精度の良い日本語対応言語モデルを作りたい。」計算資源が豊富でないのでO(N^2)であるattention機構を使わずになにか他の仕組みを使って学習させようと思います。

そこで以下の論文を組み合わせた理論を考えました。批判的に修正・改善すべき点があれば教えて下さい。

[Wave Network: An Ultra-Small Language Model](https://arxiv.org/html/2411.02674v4)
[Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/html/2411.09009v1)
[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
[From Word Models to World Models: Translating from Natural Language to the Probabilistic Language of Thought](https://arxiv.org/pdf/2306.12672)
[Large Language Diffusion Models](https://arxiv.org/pdf/2306.12672)
[WaveletGPT: Wavelets Meet Large Language Models](https://arxiv.org/abs/2409.12924)
[WavSpA: Wavelet Space Attention for Boosting Transformers’ Long Sequence Learning Ability](https://arxiv.org/pdf/2210.01989)


データはとりあえず小規模な実験から始めたく、以下のJSQuADデータセットとしたい。
# Hugging FaceからJSQuADデータセットをロードする
dataset = load_dataset("shunk031/JGLUE", name="JSQuAD")

---

**理論フレームワークの詳細な設計案**

---

## 🚩モデル設計案：『WaveHierarchicalLM（仮称）』

**全体コンセプト：**

- Wave Networkを基盤とし、多層かつ階層的に複素ベクトル表現を学習。
- Attentionは使用せず、Wave Networkの複素ベクトルベース表現にRoFormerのRoPEを統合し相対位置関係を明示化。
- WaveletGPTのWavelet変換をオプションで追加し、階層構造をモデル化。
- マスクを伴うDiffusionベースの学習で、モデルのロバスト性を強化 ?
- 効率的な語彙最適化（Cut Your Losses）とテスト時記憶機構（Titans）を導入可能に。
- DeepSpeed, mixed precision, 8bit Adamのオプション搭載。
- GRPOで事後学習:日本語QA向けのGRPO類似データセットとしては、Hugging Faceなどで公開されている「日本語推論QAセット」（例: llm-jp/ja_cot_qa や lmqg/jaquad-cotなど）を活用できます。
---

## 📌アーキテクチャ詳細（提案）

### ① Wave Networkの多層化 (Wave Encoder Block)

**目的：**  
複素数の埋め込みを使い、シーケンス内のトークン間の関係を効率的にモデル化。

- 入力トークン \( x_t \) を複素ベクトル空間 \(\mathbb{C}^d\) に埋め込み：
  \[
  x_t \in \mathbb{R}^{V} \mapsto e_t \in \mathbb{C}^{d}
  \]

- Wave Network の基本構造を層単位で繰り返し階層化：
  \[
  E^{(l+1)} = \text{WaveBlock}(E^{(l)}), \quad l=0,1,\dots,L-1
  \]

- `L` はconfigで可変に設定（デフォルトは小規模な3程度）

---

### ② RoPE (RoFormerのrotary positional embedding)による位置符号化

**目的：**  
Attentionなしで単語間の位置関係を相対的に捕捉。

- Wave Networkで得られた複素ベクトル表現に対しRoPEを適用：
  \[
  e'_t = \text{RoPE}(e_t, t)
  \]

- RoPEの位相回転による単語間の相対位置エンコーディングを活用：
  \[
  \text{RoPE}(e_t, t) = e_t \odot (\cos\theta_t + i\sin\theta_t)
  \]

- 位置符号化の位相 \(\theta_t\) はトークンの相対位置に基づき生成。

---

### ③ Wavelet変換を利用した階層構造化 (WaveletGPT由来・オプション機能)

**目的：**  
トークン埋め込みに多解像度解析を導入し、言語の階層構造を明示的にモデル化。

- Configで `use_wavelet`（bool）を設定可能（default: True）。
  - **RoPE適用直後**の埋め込みベクトルにWavelet変換を適用。
- Wavelet変換は離散ウェーブレット変換(DWT)を用いることで、多解像度な階層表現を獲得。

---

### ④ Diffusionによるトークン予測の訓練戦略 (Large Language Diffusion Models由来)

**目的：**  
トークンのノイズ除去を通じてロバストな予測を実現。

- トークン予測問題をDiffusion過程のマスク復元問題として定式化：
  - ランダムマスクを施したシーケンスの復元を通じて学習：
    \[
    \mathcal{L}_{diffusion} = \mathbb{E}_{t\sim\mathcal{U}(1,T),\epsilon\sim\mathcal{N}(0,I)} [ \| x - f_\theta(x_t, t) \|^2 ]
    \]
  - マスク率や拡散ステップ数をConfigで調整可能に。

---

### ⑤ 語彙サイズの最適化 (Cut Your Losses由来)

現在、複数の日本語対応トークナイザーが公開されています。日本語テキストの処理効率を示す指標「length per token (lpt)」で比較した場合、以下のようなランキングとなっています：この中から実際に利用可能で適切なものなものを選んでください

1. cyberagent/open-calm-7b（lpt: 2.05）
2. stockmark/gpt-neox-japanese-1.4b（lpt: 1.99）
3. rinna/bilingual-gpt-neox-4b（lpt: 1.95）
4. NovelAI/nerdstash-tokenizer-v2（lpt: 1.76）
5. google/gemma-2b（lpt: 1.59）
6. llm-jp/llm-jp-13b-v2.0（lpt: 1.51）

---

### ⑥ テスト時の記憶機構導入（Titans由来・オプション機能）

**目的：**  
テスト時に既出の系列に対する復元精度を高める。

- Configで `use_test_memory` (bool)を設定可能（default: False）。
- 学習時に外部記憶メカニズムは用いず、テスト時にのみ外部記憶を使用。

---

### ⑦ 高速化・効率化オプション

- Configで以下をon/off可能（default: False）
  - DeepSpeedによる分散・並列処理最適化
  - mixed precision (`fp16`)の有効化
  - 8bit Adam（またはLion optimizer）によるメモリ節約

---

## 📐提案する実験手順：

① **データ準備**
```python
from datasets import load_dataset

dataset = load_dataset("shunk031/JGLUE", name="JSQuAD")
```

② **基本実装ステップ**
- 複素ベクトル表現とWave Networkを実装（WaveHierarchicalLMクラス）
- RoPEによる相対位置エンコード追加
- Diffusionモデルによるマスク復元型学習を実装
- オプション設定（Wavelet, Titans）を実装

③ **評価・チューニング**
- JSQuADにてQA精度を評価し、TinySwallowをベンチマークとする
- パラメータ数は最初10M～100M前後を目安にColabのA100（40GB VRAM）範囲内で探索
- 精度・パフォーマンスを測定し、精度とリソースのバランスを調整

---

## 🧪最終目標

- Google ColabのA100 GPU一枚で効率よく訓練可能なモデル
- Attention無しでも高精度で日本語QAタスクを解ける小型モデル
- TinySwallow相当の精度（JSQuAD基準）

---

## 📂 実装構成（モジュール別整理）

```
slm/
├── config.py                 # 各種ハイパーパラメータ設定
├── tokenizer.py              # Japanse Tokenizer
├── data_downloader.py        # Google Driveでデータ保存・読込
├── data_loader.py            # データ前処理・DataLoader
├── model.py                  # WaveNet+Wavelet+RoPEモデル
├── cce_loss.py               # CCE損失の実装
├── train.py                  # 事前学習＋GRPO Fine-tuning
├── evaluate.py               # テストデータ評価
├── inference.py              # 推論詳細確認
└── compareLM.py              # tiny-lmとの精度比較評価

+ test code
...

```

保存は　"/content/drive/MyDrive/data", "/content/drive/MyDrive/weights" などフォルダを作って保存してください。



