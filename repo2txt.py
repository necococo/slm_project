import os

# 出力ファイル名
output_file = "all_code_combined.txt"

# 対象ディレクトリ
target_dir = "."

# 無視するファイルやディレクトリ
ignored_dirs = {".git", "__pycache__", ".slm", "tests", "memo", ".pytest_cache"}
ignored_files = {output_file, "repo2txt.py", "copilot_code_style.md", ".gitignore", }

# ファイルがテキストかを確認する関数
def is_text_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            f.read(1024)
        return True
    except:
        return False

# 出力ファイルを作成
with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk(target_dir):
        # 無視するディレクトリを除外
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        # フォルダ名を出力
        outfile.write(f"--- Directory: {root} ---\n\n")
        for file in files:
            if file in ignored_files:
                continue
            file_path = os.path.join(root, file)
            # テキストファイルのみ処理
            if is_text_file(file_path):
                try:
                    outfile.write(f"--- File: {file_path} ---\n")
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n\n")
                except Exception as e:
                    outfile.write(f"Error reading {file_path}: {e}\n")
            else:
                outfile.write(f"Skipping binary file: {file_path}\n")