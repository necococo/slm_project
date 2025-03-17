#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
How:
    評価用のスクリプト。Googleドライブからデータを高速アクセス可能な
    ローカルディレクトリにコピーします。

Why not:
    Googleドライブは直接アクセスすると遅いため、高速なローカルストレージに
    データをコピーして処理速度を向上させます。
"""

import os
import shutil
from pathlib import Path
import time
from typing import Optional


def copy_data_to_fast_storage(
    source_path: str = "/content/drive/MyDrive/slm/data/fujiki/wiki40b_ja",
    target_path: str = "/content/fast_data/",
) -> None:
    """
    Googleドライブからローカルの高速ストレージにデータをコピーします。

    Args:
        source_path: コピー元のパス
        target_path: コピー先のパス
    
    Returns:
        None
    """
    # 開始時間を記録
    start_time = time.time()
    
    # ターゲットディレクトリが存在しない場合は作成
    target_dir = Path(target_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # コピー先のフルパス
    full_target_path = target_dir / Path(source_path).name
    
    # コピー元が存在するか確認
    if not Path(source_path).exists():
        raise FileNotFoundError(f"コピー元のパスが見つかりません: {source_path}")
    
    # すでにコピー先にデータがある場合はスキップするオプションを提供
    if full_target_path.exists():
        print(f"コピー先にデータがすでに存在します: {full_target_path}")
        return
    else:
        shutil.copytree(source_path, full_target_path)
        print(f"データをコピーしました: {source_path} -> {full_target_path}")        



if __name__ == "__main__":
    try:
        copy_data_to_fast_storage()
    except Exception as e:
        print(f"スクリプト実行中にエラーが発生しました: {str(e)}")
