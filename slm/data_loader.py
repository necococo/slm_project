# # data_loader.py
# import os
# from typing import Optional, Dict, Any, Tuple, Union
# from datasets import load_dataset, Dataset, DatasetDict
# from transformers import AutoTokenizer

# def tokenize_fn(example: Dict[str, Any], tokenizer, max_len: int) -> Dict[str, Any]:
#     """
#     How:
#         - QAの場合は question + context + answers をまとめる
#         - なければ example["text"] を使う
#         - Hugging FaceトークナイザでMLM用に 'input_ids','attention_mask','labels'を作る
#           (labelsは別途collatorで設定可能なので、ここではinput_idsだけでもよい)
#     """
#     if "question" in example and "context" in example and "answers" in example:
#         text = f"質問: {example['question']} 文脈: {example['context']} 答え: {example['answers']['text'][0]}"
#     elif "text" in example:
#         text = example["text"]
#     else:
#         text = str(example)

#     # トークナイザ呼び出し
#     encoded = tokenizer(
#         text,
#         truncation=True,
#         max_length=max_len,
#         # special_tokens => [CLS], [SEP] など自動付与
#         return_special_tokens_mask=True
#     )
#     return encoded


# def load_preprocessed_dataset(
#     dataset_name: str,
#     dataset_subset: Optional[str],
#     cache_dir: str,
#     tokenizer,
#     max_len: int = 512
# ) -> Tuple[Union[Dataset, None], Union[Dataset, None]]:
#     """
#     1. Hugging Faceからデータセットをロード
#     2. map(tokenize_fn) で前処理
#     3. train/validationを取り出して返す
#     """
#     print(f"=== データセット {dataset_name}/{dataset_subset} をロード＆前処理 ===")
#     ds = load_dataset(dataset_name, dataset_subset, cache_dir=cache_dir)

#     ds = ds.map(lambda ex: tokenize_fn(ex, tokenizer, max_len), batched=False)

#     # train/validation 分割を探す
#     if isinstance(ds, dict):  # DatasetDict
#         train_ds = ds.get("train", None)
#         valid_ds = ds.get("validation", ds.get("dev", None))
#     else:
#         train_ds = ds
#         valid_ds = None

#     return train_ds, valid_ds


