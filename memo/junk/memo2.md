これまでの議論を踏まえ、戦略を再整理します。

---

## 🚩 戦略の整理（再確認用）

### ✅ 最終的なモデルの目的

- **A100 GPU 1枚で効率的に動作する日本語対応の小型高性能LM**
- 日本語QAタスク（JSQuADなど）で精度が良く、CoT（推論能力）も持つことを目指す

---

### 🧩 採用技術の再確認

| 採用する技術 | 理由・目的 |
|--------------|-----------|
| Wave Network (複素ベクトル) | Attentionなしで効率的に単語・文脈関係を捉える |
| Wavelet変換（逆変換なし）| 階層的な情報圧縮を効率よく実現 |
| RoPE（回転位置エンコーディング）| トークンの相対位置関係を効率的にモデル化 |
| GRPO(CoT) Fine-tuning| 推論過程の明示化・論理的推論力の向上 |
| Cut Cross Entropy(CCE)| 語彙サイズを最適化し計算コストを削減 |
| Japanese Tokenizer | 日本語に最適化されたトークナイザー |

---

### 🚫 採用しない技術（複雑さや計算負荷が理由）

- Diffusion（高計算コスト）
- Titans記憶機構（当面必要なし）
- 逆Wavelet変換（復元精度のリスクと追加コストのため省略）

---

## 📂 実装構成（モジュール別整理）

```
slm/
├── config.py                 # 各種ハイパーパラメータ設定
├── tokenizer.py              # Japanse Tokenizer
├── data_downloader.py        # Google Driveでデータ保存・読込
├── data_loader.py            # データ前処理・DataLoader
├── data_loader_test.py       # データローダーテスト
├── model.py                  # WaveNet+Wavelet+RoPEモデル
├── model_test.py             # モデルの単体テスト
├── cce_loss.py               # CCE損失の実装
├── cce_loss_test.py          # CCE損失のテストコード
├── train.py                  # 事前学習＋GRPO Fine-tuning
├── evaluate.py               # テストデータ評価
├── inference.py              # 推論詳細確認
└── compareLM.py              # tiny-lmとの精度比較評価
```

---

## 📌 データ前処理方針（最終決定）

- **novelai Tokenizerを採用**
  ```python
  import sentencepiece as spm
  tokenizer = spm.SentencePieceProcessor(model_file="/content/drive/MyDrive/tokenizer/novelai.model")
  ```

---

## 🧪 評価対象モデル（比較対象）

- 小型日本語LLMとして`sbintuitions/tiny-lm`を基準に精度比較する。

---

## 🚀 トレーニング・評価手順（まとめ）

① **データ準備**
- JSQuADで事前学習、GRPOベースの日本語CoT QAデータセットでFine-tuning。

② **モデル学習**
- Wave Network（複素ベクトル）+ Wavelet + RoPEモデルを構築・事前学習。
- Fine-tuningでCoT（推論能力）を強化。

③ **評価**
- JSQuADの検証セットで性能評価。
- tiny-lmとの性能比較を行う。

④ **推論詳細確認**
- `inference.py`を用いて推論の中身（CoTの論理推論経路など）を確認する。

---

## ✅ 次の作業（次に提示するもの）

- 上記モジュール（config.pyから順に）の実装を開始。
- pytestでのテストコードも各モジュールに添付して提示。

以上で問題ないか再確認をお願いします。