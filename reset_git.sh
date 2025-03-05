#!/bin/bash

# 現在のディレクトリを確認
current_dir=$(pwd)
echo "現在のディレクトリ: $current_dir"

# リポジトリのルートディレクトリに移動
cd /Users/kj/slm || { echo "Error: ディレクトリが存在しません"; exit 1; }

# 現在のリポジトリをバックアップ
backup_dir="../slm_backup_$(date +%Y%m%d%H%M%S)"
echo "リポジトリをバックアップ: $backup_dir"
cp -r . "$backup_dir"

# Gitリポジトリを初期化
echo "Gitリポジトリを初期化します..."
rm -rf .git
git init

# .gitignoreファイルを作成
cat > .gitignore << EOL
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.slm/

.DS_Store

all_code_combined.txt
copilot_code_style.md
EOL

# すべてのファイルをステージング
git add .

# 初期コミット
git commit -m "Initial commit: Wave Network Language Model"

# 確認メッセージ
echo "Gitリポジトリを初期化し、現在のコードベースを初期コミットとしました。"
echo "リモートリポジトリを設定するには以下のようなコマンドを実行してください："
echo "git remote add origin <repository-url>"
echo "git push -u origin main"
