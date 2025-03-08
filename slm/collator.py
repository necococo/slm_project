# filename: collator.py

from typing import Dict, Optional, Tuple, Union
import torch

class CustomCollator:
    """
    How:
        - トークン列を DataLoader から受け取り、最大シーケンス長 (config.ModelConfig.max_seq_len)
          を超えるものはトリミング、満たない場合はパディングする。
        - MLM の場合はマスク処理を行う。
        - QA タスクの場合はQAラベルを処理する（今回省略）。
    """
    def __init__(
        self,
        tokenizer,
        model_config,  # ModelConfig を渡す
        mlm: bool = False,
        mlm_probability: float = 0.15,
        mask_token_id: Optional[int] = None,
        qa: bool = False,
        dynamic_padding: bool = True
    ):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.qa = qa
        self.dynamic_padding = dynamic_padding
        self.warning_shown = False

        if mask_token_id is not None:
            self.mask_token_id = mask_token_id
        else:
            if tokenizer.mask_token_id is not None:
                self.mask_token_id = tokenizer.mask_token_id
            else:
                tmp_id = tokenizer.convert_tokens_to_ids("[MASK]")
                if tmp_id == tokenizer.unk_token_id:
                    print("[WARNING] [MASK] is not in the vocab. Using unk_token_id as mask.")
                    self.mask_token_id = tokenizer.unk_token_id
                else:
                    self.mask_token_id = tmp_id

    def mask_tokens(self, inputs, mask_token_id=None):
        """
        入力トークンの一部をマスクするメソッド
        mask_token_idが明示的に提供されない場合、代替手段としてunk_token_idを使用
        """
        if mask_token_id is None:
            # tokenizer.mask_token_idがない場合、unk_token_idを使用
            mask_token_id = self.tokenizer.unk_token_id
            if not self.warning_shown:
                print(f"マスクトークンIDがないため、代わりにunk_token_id ({mask_token_id}) を使用します")
                self.warning_shown = True
        
        # ...masking logic...

    def __call__(self, examples):
        # バッチ内の入力を収集
        input_ids = [e["input_ids"] for e in examples]
        attention_mask = [e["attention_mask"] if "attention_mask" in e else [1] * len(e["input_ids"]) for e in examples]
        
        # シーケンス長を制限（model_configのmax_seq_lenまで）
        max_len = self.model_config.max_seq_len
        input_ids = [ids[:max_len] for ids in input_ids]
        attention_mask = [mask[:max_len] for mask in attention_mask]
        
        # パディング処理
        padded_input_ids = []
        padded_attention_mask = []
        for ids, mask in zip(input_ids, attention_mask):
            padding_length = max_len - len(ids)
            if padding_length > 0:
                padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * padding_length)
                padded_attention_mask.append(mask + [0] * padding_length)
            else:
                padded_input_ids.append(ids)
                padded_attention_mask.append(mask)
        
        # MLMタスク用のマスクを適用（必要に応じて）
        if self.mlm:
            probability_matrix = torch.full((len(padded_input_ids), max_len), self.mlm_probability)
            special_ids = set([
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.unk_token_id,
                self.tokenizer.pad_token_id
            ])
            mask_arr = torch.bernoulli(probability_matrix).bool()
            for s_id in special_ids:
                mask_arr &= (torch.tensor(padded_input_ids) != s_id)
            
            input_ids_batch = torch.tensor(padded_input_ids)
            labels_batch = input_ids_batch.clone()
            input_ids_batch[mask_arr] = self.mask_token_id
            labels_batch[~mask_arr] = -100
        else:
            input_ids_batch = torch.tensor(padded_input_ids)
            labels_batch = input_ids_batch.clone()

        # テンソルに変換して返す
        return {
            "input_ids": input_ids_batch,
            "attention_mask": torch.tensor(padded_attention_mask),
            "labels": labels_batch
        }