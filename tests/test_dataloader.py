# test_dataloader.py
import os
import pickle
import torch
from slm.data_loader import WikiDataset

def test_wiki_dataset(tmp_path):
    # テスト用にダミーのデータを作成
    dummy_tokens = [
        list(range(50)),  # サンプル1
        list(range(30))   # サンプル2
    ]
    file_path = tmp_path / "dummy.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(dummy_tokens, f)
    
    seq_length = 40
    dataset = WikiDataset(str(file_path), seq_length)
    assert len(dataset) == 2, "Dataset length should be 2"
    sample = dataset[0]
    # "input_ids"と"labels"が存在するか
    assert "input_ids" in sample and "labels" in sample, "Keys 'input_ids' and 'labels' must be in the sample"
    # 長さが固定長になっていることを確認
    assert sample["input_ids"].shape[0] == seq_length, "Sequence length mismatch"