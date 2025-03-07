~/slm/wavelet_transformer/
├── config.py                   # モデル設定と学習設定
├── models/
│   ├── wavelet_model.py        # モデルA: wavelet+SVD+attention
│   ├── transformer_model.py    # モデルB: 標準Transformer
│   └── components/             # モデルコンポーネント
│       ├── wavelet_layer.py    # Wavelet変換実装
│       ├── svd_attention.py    # SVD低ランク注意機構
│       └── linear_attention.py # 線形計算量注意機構
├── training/
│   ├── trainer.py              # 学習ループ
│   ├── metrics.py              # 評価指標
│   └── losses.py              # カスタム損失関数(Cut Cross Entropy)
├── data/
│   ├── dataset.py              # データセット処理
│   └── preprocessor.py         # データ前処理
├── utils/
│   ├── visualization.py        # 可視化ユーティリティ
│   └── common.py               # 共通関数
├── train_wavelet.py            # Waveletモデル学習スクリプト
└── train_transformer.py        # Transformerモデル学習スクリプト