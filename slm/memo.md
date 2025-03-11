# tokenizer
tokenizer_name: str = "cl-tohoku/bert-base-japanese-whole-word-masking", # Whole Word Masking: ある単語の一部だけがマスクされ不自然な断片が残らないよう、連続するサブワードをまとめて一緒にマスクします。

# data download


import pandas as pd    
from datasets import load_dataset
# dataset = load_dataset("json", data_files=["file1.json", "file2.json", "file3.json"])
dataset = load_dataset("json", data_files="xxxx.json")


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-1b", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-1b")