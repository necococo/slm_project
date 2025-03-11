Optunaの原理と使い方

  Optunaは効率的なハイパーパラメータ最適化フレームワークで、機械学習モデルの性能を向上させるために使用されます。

  基本原理

  1. ベイズ最適化:
  過去の試行結果に基づいて次に試すパラメータを賢く選択します。これにより、無作為な探索よりも効率よく最適なパラメータを見つけられます。
  2. 木構造Parzen推定器 (TPE): デフォルトのサンプラーとして使用され、過去のパラメータを「良い結果を出した」と「悪い結果を出した」に分
  類し、良い結果を出した領域を優先的に探索します。
  3. 動的な探索空間: 条件付きパラメータの定義が可能で、他のパラメータ値に依存した探索空間を設定できます。
  4. 早期打ち切り: 性能の悪い試行を早期に中止し、計算リソースを節約します。

  基本的な使い方

  import optuna

  # 目的関数の定義
  def objective(trial):
      # パラメータのサンプリング
      x = trial.suggest_float('x', -10, 10)
      y = trial.suggest_float('y', -5, 5)

      # 最小化したい値を返す
      return (x - 2) ** 2 + (y - 3) ** 2

  # Studyの作成と最適化の実行
  study = optuna.create_study(direction='minimize')
  study.optimize(objective, n_trials=100)

  # 結果の表示
  print(f"Best params: {study.best_params}")
  print(f"Best value: {study.best_value}")

  主要な機能

  1. パラメータのサンプリング方法:
    - suggest_float: 連続値
    - suggest_int: 整数値
    - suggest_categorical: カテゴリカル値
    - suggest_loguniform: 対数スケールでの連続値
  2. 可視化ツール:
    - パラメータの重要度
    - 最適化の履歴
    - パラメータの関係性の可視化
  3. 並列・分散実行:
    - 複数のプロセスやマシンで並列に探索できます
    - RDBやRedisなどによるストレージバックエンド対応
  4. プルーニング機能:
    - MedianPruner: 中央値より悪い試行を打ち切り
    - PercentilePruner: 特定パーセンタイルより悪い試行を打ち切り

  実践的な使い方

  1. 機械学習モデルの最適化:
  def objective(trial):
      # モデルのハイパーパラメータをサンプリング
      n_layers = trial.suggest_int('n_layers', 1, 5)
      learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

      # モデルを構築・訓練
      model = MyModel(n_layers=n_layers)
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

      # 訓練と評価
      train_model(model, optimizer)
      validation_error = evaluate_model(model)

      return validation_error

  study = optuna.create_study()
  study.optimize(objective, n_trials=50)

  2. 結果の保存と再開:
  # 結果の保存
  study = optuna.create_study(study_name='my_study', storage='sqlite:///optuna.db')
  study.optimize(objective, n_trials=10)

  # 後で再開
  resumed_study = optuna.load_study(study_name='my_study', storage='sqlite:///optuna.db')
  resumed_study.optimize(objective, n_trials=10)