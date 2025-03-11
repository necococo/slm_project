import os
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from datasets import load_dataset
import pytest


class WikiDataset(Dataset):
    """
    Wikipediaデータセット用のPyTorch Datasetクラス。
    各サンプルはトークンIDのリストです。
    メモリ効率化モードでは全データを読み込まず、必要な行だけを読み込みます。
    """
    def __init__(self, file_path: str, tokenizer: spm.SentencePieceProcessor, max_length: int = 512, 
                 memory_efficient: bool = False, max_samples: Optional[int] = None) -> None:
        """
        :param file_path: テキストファイルのパス
        :param tokenizer: 学習済みSentencePieceProcessorのインスタンス
        :param max_length: 最大トークン数（長い場合は切り捨てる）
        :param memory_efficient: メモリ効率化モード（大きなデータセット向け）
        :param max_samples: 最大サンプル数（None=全て使用）
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.memory_efficient = memory_efficient
        
        if not memory_efficient:
            # 従来の方法：全データをメモリに読み込む
            print(f"データセットをメモリに読み込み中: {file_path}")
            import time
            start_time = time.time()
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.lines = [line for line in f.readlines() if line.strip()]
                    if max_samples is not None:
                        self.lines = self.lines[:max_samples]
            except UnicodeDecodeError:
                # UTF-8デコードに失敗した場合はバイナリモードで開き直す
                print(f"警告: UTF-8デコードに失敗しました。バイナリモードで再試行します。")
                with open(file_path, "rb") as f:
                    self.lines = [line.decode('utf-8') for line in f.readlines() if line.strip()]
                    if max_samples is not None:
                        self.lines = self.lines[:max_samples]
            load_time = time.time() - start_time
            print(f"データ読み込み完了: {len(self.lines)}行, {load_time:.2f}秒")
        else:
            # メモリ効率化モード：行数だけカウントし、インデックスを作成
            self.lines = None
            self.line_offsets = []
            print(f"メモリ効率化モードでデータセットをインデックス化中: {file_path}")
            start_time = time.time()
            with open(file_path, "r", encoding="utf-8") as f:
                offset = 0
                sample_count = 0
                for line in f:
                    if line.strip():  # 空行はスキップ
                        self.line_offsets.append(offset)
                        sample_count += 1
                        if max_samples is not None and sample_count >= max_samples:
                            break
                    offset += len(line.encode('utf-8'))  # バイトオフセットを使用
            index_time = time.time() - start_time
            print(f"インデックス化完了: {len(self.line_offsets)}行, {index_time:.2f}秒")

    def __len__(self) -> int:
        if self.memory_efficient:
            return len(self.line_offsets)
        else:
            return len(self.lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.memory_efficient:
            # オンデマンドで特定の行を読み込む
            text = self._read_line_at_offset(self.line_offsets[idx]).strip()
        else:
            text = self.lines[idx].strip()
        
        # バイト文字列が混入していないか確認して対処
        if isinstance(text, bytes):
            try:
                text = text.decode('utf-8')
                print(f"警告: バイト文字列を検出しデコードしました (idx={idx})")
            except UnicodeDecodeError as e:
                print(f"エラー: バイト文字列のデコードに失敗しました (idx={idx}): {e}")
                text = ""
        
        # テキストをトークン化
        token_ids: List[int] = self.tokenizer.encode(text, out_type=int)
        # max_length以上は切り捨て
        token_ids = token_ids[:self.max_length]
        return {"input_ids": torch.tensor(token_ids, dtype=torch.long)}
    
    def _read_line_at_offset(self, offset: int) -> str:
        """ファイルの特定のオフセットから1行読み込む"""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                f.seek(offset)
                line = f.readline()
                # バイト文字列が混入していないか確認
                if isinstance(line, bytes):
                    line = line.decode('utf-8')
                return line
        except UnicodeDecodeError:
            # UTF-8デコードエラーが出た場合はバイナリモードで開き直す
            try:
                with open(self.file_path, "rb") as f:
                    f.seek(offset)
                    return f.readline().decode('utf-8')
            except Exception as e:
                print(f"エラー: ファイル読み込みに失敗しました: {e}")
                return ""


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    ミニバッチ内のデータをパディングするcollate関数。
    Diffusionモデル学習と互換性を持つように拡張。
    
    How: torch.nn.utils.rnn.pad_sequenceを利用
    Why not: バッチ内のシーケンス長を揃えるため
     
    :param batch: 各サンプルの辞書リスト
    :return: パディング済みのinput_idsを含む辞書
    """
    # 入力シーケンスを取得（WikiDataset.__getitem__ですでに最大長を切り詰め済み）
    input_ids = [item["input_ids"] for item in batch]
    
    # シーケンスをパディング
    padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    
    # attention_maskを作成
    attention_mask = torch.zeros_like(padded, dtype=torch.long)
    for i, ids in enumerate(input_ids):
        attention_mask[i, :len(ids)] = 1
    
    # diffusionモデル学習用にlabelsを入力と同じに設定
    # (実際のマスキングはdiffusionモデルが行う)
    return {
        "input_ids": padded,
        "attention_mask": attention_mask,
        "labels": padded.clone()  # labelsをinput_idsと同じに設定
    }


def load_tokenizer(model_dir: str, model_prefix: str) -> spm.SentencePieceProcessor:
    """
    学習済みSentencePieceモデルをロードする関数。
    
    :param model_dir: モデルファイルが保存されているディレクトリのパス
    :param model_prefix: モデルファイルの接頭辞（例: "sp_jwiki"）
    :return: SentencePieceProcessorのインスタンス
    """
    model_path: str = os.path.join(model_dir, f"{model_prefix}.model")
    sp: spm.SentencePieceProcessor = spm.SentencePieceProcessor(model_file=model_path)
    return sp


def test_tokenizer_functionality(tokenizer: spm.SentencePieceProcessor, text: str) -> None:
    """
    トークナイザの機能チェック用のテスト関数。
    
    How: 指定したテキストをエンコード、デコードして結果を表示
    Why not: 直接の言語モデル学習前に、トークン化の動作を確認するため
     
    :param tokenizer: 学習済みSentencePieceProcessorのインスタンス
    :param text: テストするテキスト文字列
    """
    # トークンID取得
    tokens_ids: list[int] = tokenizer.encode(text, out_type=int)
    # 文字列トークン取得
    tokens_str: list[str] = tokenizer.encode(text, out_type=str)
    # 復元
    decoded_text: str = tokenizer.decode(tokens_ids)
    
    print("元の文章:", text)
    print("トークンID:", tokens_ids)
    print("トークン化結果:", tokens_str)
    print("復元結果:", decoded_text)
    
    # 完全一致するか確認
    if text == decoded_text:
        print("✓ 完全一致しました")
    else:
        print("× 不一致があります")
        # 差分を視覚的に表示
        for i, (orig, dec) in enumerate(zip(text, decoded_text)):
            if orig != dec:
                print(f"  位置 {i}: 元の文字「{orig}」 → 復元後「{dec}」")


def clean_text(batch: dict) -> dict:
    """
    テキスト内のプレースホルダートークンを削除する関数。
    
    How: 文字列の置換を利用して不要なトークンを除去
    Why not: マニュアルでの再前処理を避けるため
    """
    text: str = batch["text"]
    for token in ["_START_ARTICLE_", "_START_SECTION_", "_START_PARAGRAPH_"]:
        text = text.replace(token, "")
    text = text.replace("_NEWLINE_", "\n")
    batch["text"] = text
    return batch


def prepare_dataset(data_dir: str) -> tuple:
    """
    Wiki-40B 日本語データセットを前処理し、テキストファイルに保存する関数
    
    :param data_dir: データを保存するディレクトリパス
    :return: 各スプリットのファイルパスを含むタプル
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Wiki-40B 日本語データセットの全スプリットを取得する
    datasets = load_dataset("wiki40b", "ja")
    train_ds = datasets["train"]
    valid_ds = datasets["validation"]
    test_ds = datasets["test"]
    
    # 各スプリットに対して前処理を実施
    train_ds = train_ds.map(clean_text)
    valid_ds = valid_ds.map(clean_text)
    test_ds = test_ds.map(clean_text)
    
    # 各スプリットを別々のテキストファイルに保存する
    train_path: str = os.path.join(data_dir, "wiki40b_ja_train.txt")
    valid_path: str = os.path.join(data_dir, "wiki40b_ja_valid.txt")
    test_path: str = os.path.join(data_dir, "wiki40b_ja_test.txt")
    
    with open(train_path, "w", encoding="utf-8") as f_train:
        for record in train_ds:
            f_train.write(record["text"] + "\n")
    
    with open(valid_path, "w", encoding="utf-8") as f_valid:
        for record in valid_ds:
            f_valid.write(record["text"] + "\n")
    
    with open(test_path, "w", encoding="utf-8") as f_test:
        for record in test_ds:
            f_test.write(record["text"] + "\n")
    
    return train_path, valid_path, test_path


def train_tokenizer(train_path: str, valid_path: str, data_dir: str, model_prefix: str = "sp_jwiki", vocab_size: int = 32000) -> None:
    """
    SentencePieceトークナイザを訓練する関数
    
    :param train_path: 訓練データのファイルパス
    :param valid_path: 検証データのファイルパス
    :param data_dir: モデルを保存するディレクトリパス
    :param model_prefix: モデルファイルの接頭辞
    :param vocab_size: 語彙サイズ
    """
    import multiprocessing
    import shutil
    import time
    
    # Google Colab環境の検出
    is_colab = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False
    
    # trainとvalidを統合
    combined_path: str = os.path.join(data_dir, "wiki40b_ja_train_valid.txt")
    print(f"訓練データとテストデータを統合中... 保存先: {combined_path}")
    with open(combined_path, "w", encoding="utf-8") as f_combined:
        for path in [train_path, valid_path]:
            print(f"処理中: {path}")
            with open(path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    if line.strip():  # 空行をスキップ
                        f_combined.write(line)
    
    # ファイルサイズを表示
    file_size_mb = os.path.getsize(combined_path) / (1024 * 1024)
    print(f"統合ファイルサイズ: {file_size_mb:.2f} MB")
    
    # 使用するスレッド数を制限（Colabではリソース制約を考慮）
    num_threads = min(4, multiprocessing.cpu_count())
    print(f"SentencePieceトレーニングに{num_threads}スレッドを使用します")
    
    # Colabでは処理を最適化
    local_combined_path = combined_path
    local_model_prefix = os.path.join(data_dir, model_prefix)
    
    if is_colab and "/content/drive/" in combined_path:
        print("Google Colabで実行中: ローカルストレージを利用して処理を最適化します")
        # ローカルにファイルをコピー
        local_combined_path = "/content/wiki40b_ja_train_valid.txt"
        local_model_prefix = "/content/sp_jwiki"
        
        print(f"Driveから/contentにファイルをコピー中... ファイルサイズ: {file_size_mb:.2f} MB")
        start_time = time.time()
        shutil.copy(combined_path, local_combined_path)
        copy_time = time.time() - start_time
        print(f"コピー完了! 所要時間: {copy_time:.2f}秒")
    
    print("SentencePieceモデルのトレーニングを開始します...")
    start_time = time.time()
    
    # SentencePieceトレーニングの実行
    spm.SentencePieceTrainer.Train(
        f"--input={local_combined_path} "
        f"--model_prefix={local_model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--user_defined_symbols=[MASK] "
        f"--num_threads={num_threads} "
        f"--input_sentence_size=1000000 "  # サンプルとして100万文を利用
        f"--shuffle_input_sentence=true "  # 入力文をシャッフルして偏りを軽減
        f"--character_coverage=0.9995 "    # 日本語に最適な設定
        f"--max_sentence_length=4192"      # 長すぎる文を防止
    )
    
    train_time = time.time() - start_time
    print(f"トレーニング完了! 所要時間: {train_time:.2f}秒")
    
    # Colabで実行時、モデルをDriveにコピー
    if is_colab and "/content/drive/" in combined_path:
        print(f"生成されたモデルをGoogle Driveにコピー中...")
        for ext in [".model", ".vocab"]:
            src = local_model_prefix + ext
            dst = os.path.join(data_dir, model_prefix + ext)
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"コピー完了: {dst}")
        
        # ローカルファイルを削除して容量節約
        for ext in [".model", ".vocab"]:
            local_file = local_model_prefix + ext
            if os.path.exists(local_file):
                os.remove(local_file)
        
        if os.path.exists(local_combined_path):
            os.remove(local_combined_path)
            print("一時ファイルをクリーンアップしました")
            
    print(f"トークナイザーモデルが保存されました: {os.path.join(data_dir, model_prefix)}.model")


def test_load_tokenizer():
    """トークナイザのロードをテストする関数"""
    data_dir = "/content/drive/MyDrive/slm/data/fujiki/wiki40b_ja"
    model_prefix = "sp_jwiki"
    tokenizer = load_tokenizer(data_dir, model_prefix)
    assert isinstance(tokenizer, spm.SentencePieceProcessor)
    
    sample_text = "テスト文章です。"
    tokens = tokenizer.encode(sample_text, out_type=int)
    assert isinstance(tokens, list)
    decoded_text = tokenizer.decode(tokens)
    assert isinstance(decoded_text, str)
    # エンコードとデコード後、元のテキストに近い内容になっているか確認（完全一致は前処理次第）
    assert len(decoded_text) > 0


def create_dataloader(dataset_path: str, tokenizer: spm.SentencePieceProcessor, 
                     batch_size: int = 32, max_length: int = 512, 
                     memory_efficient: bool = False, 
                     max_samples: Optional[int] = None,
                     num_workers: int = 0,
                     pin_memory: bool = False,
                     shuffle: bool = True) -> DataLoader:
    """
    データローダーを作成する関数
    
    :param dataset_path: データセットのファイルパス
    :param tokenizer: トークナイザ
    :param batch_size: バッチサイズ
    :param max_length: 最大シーケンス長
    :param memory_efficient: メモリ効率化モードを使用するか
    :param max_samples: 使用する最大サンプル数
    :param num_workers: データロードに使用するワーカー数
    :param pin_memory: GPUメモリを使用する場合はTrueに設定
    :param shuffle: データシャッフルを行うか
    :return: DataLoaderインスタンス
    """
    # Google Colab環境の検出
    is_colab = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False
    
    # Colabの場合は自動的に設定を調整
    if is_colab:
        print("Google Colab環境を検出しました - データローダー設定を最適化します")
        # Colabの場合はバッチサイズを小さくする
        batch_size = min(batch_size, 16)
        # メモリ効率化モードを推奨
        if not memory_efficient and os.path.getsize(dataset_path) > 100 * 1024 * 1024:  # 100MB以上
            print(f"大きなデータセットに対してメモリ効率化モードを自動的に有効化します（{os.path.getsize(dataset_path) / (1024 * 1024):.2f}MB）")
            memory_efficient = True
        # ワーカー数も制限
        num_workers = min(num_workers, 2)
    
    # データセットの作成
    dataset = WikiDataset(
        file_path=dataset_path, 
        tokenizer=tokenizer, 
        max_length=max_length,
        memory_efficient=memory_efficient,
        max_samples=max_samples
    )
    
    print(f"DataLoaderを作成: batch_size={batch_size}, num_workers={num_workers}, " 
          f"pin_memory={pin_memory}, データセットサイズ={len(dataset)}")
    
    return DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


if __name__ == "__main__":
    # Google Drive上の保存先ディレクトリ（適宜変更してください）
    data_dir: str = "/content/drive/MyDrive/slm/data/fujiki/wiki40b_ja"
    model_prefix: str = "sp_jwiki"
    
    # データセットの準備（必要な場合のみ実行）
    # train_path, valid_path, test_path = prepare_dataset(data_dir)
    # train_tokenizer(train_path, valid_path, data_dir)
    
    # 学習済みトークナイザのロード
    tokenizer: spm.SentencePieceProcessor = load_tokenizer(data_dir, model_prefix)
    
    # 機能チェック用のサンプルテキスト
    sample_text: str = "これはトークナイザの機能を確認するためのテスト文章です。"
    test_tokenizer_functionality(tokenizer, sample_text)
    
    # WikiDatasetの作成と動作確認
    train_file: str = os.path.join(data_dir, "wiki40b_ja_train.txt")
    dataloader = create_dataloader(train_file, tokenizer)
    
    # DataLoaderから1バッチ取得してshapeを確認
    for batch in dataloader:
        print("バッチのinput_idsのshape:", batch["input_ids"].shape)
        break
    
    # テストの実行（必要な場合）
    # pytest.main([__file__])