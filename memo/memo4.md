【フォルダ構成】

・プロジェクトルートディレクトリ：/content/slm # at google colab 

- pyproject.toml（プロジェクト全体のビルド・依存関係設定ファイル）

- slm/（パッケージフォルダ。ソースコードがここに格納される。例：data_loader.py、config.py、…）

- tests/（テストコード。init.py は空で問題ない）

- memo/（その他ドキュメント・画像等）

【pyproject.toml の内容】

・[build-system] セクション

- requires = ["setuptools>=75", "wheel"]

Why: 現在の環境（setuptools 75.1.0）に合わせ、最低限このバージョン以上を要求する。

- build-backend = "setuptools.build_meta"

・[project] セクション

- name, version, description, readme, requires-python, authors, dependencies などを定義

※ライセンスは今回は不要とし、設定から除外。

How: ここに記述された情報が、pip などのパッケージマネージャに利用される。

・[tool.setuptools] セクション

- packages = ["slm"]

- [tool.setuptools.package-dir] は、プロジェクトルート直下の「slm」フォルダが対象になるように設定するか、このセクション自体を削除

Why: 誤って "slm/slm" を参照しないようにするため。

【今回の主な対応内容と注意点】

パッケージディレクトリ設定の見直し

- 初めは "from .config" と相対インポートする設定で進め、正しいフォルダ構成（/content/slm/slm ではなく /content/slm）になるように調整。

pyproject.toml の設定調整

- setuptools のバージョン要求を現在の環境に合わせ「>=75」と設定。

- package-dir の指定が誤っていたため、正しくパッケージが認識されるよう修正（削除または "" = "." のようにルート直下を指定）。

editable install（pip install -e .）でのビルドとインストール

- editable install により、ソースコードの変更が即時反映されるようにセットアップ。

- エラー原因となっていた部分（例: package directory の指定）を修正。

【まとめ】

正しいフォルダ構成と pyproject.toml の設定により、パッケージ "slm" が正しく認識されるようになり、import エラーが解消されました。
Editable install を利用して、開発中にコード変更が即座に反映される環境を実現。
今回の調整で、setup.py を削除し、pyproject.toml による PEP 517/660 に対応する形になっています。
