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
        qa: bool = False
    ):
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.qa = qa

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

    def __call__(self, batch):
        """
        How:
            - batch(list[dict]) から input_ids / labels を取り出し、
              ModelConfig.max_seq_len でトリミング or パディング。
            - mlm=True の場合は確率的にマスクトークンに置換し、labelsを-100で埋める。
        """
        # Why not: バッチ内で統一的に max_seq_len を取得して使う
        max_len = self.model_config.max_seq_len
        
        input_ids_list = []
        
        for sample in batch:
            # 取り出し
            input_ids = sample["input_ids"]
            # QAなどがある場合は別の処理かもしれないが、とりあえず同様にラベル扱い
            
            # 1) トリミング
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]

            input_ids_list.append(input_ids)
        
        # 2) パディング
        #   バッチ内の max_len は model_config.max_seq_len と固定
        for i in range(len(input_ids_list)):
            diff = max_len - len(input_ids_list[i])
            if diff > 0:
                # 短い場合はパディング
                input_ids_list[i] += [self.tokenizer.pad_token_id] * diff

        # list -> tensor
        input_ids_batch = torch.tensor(input_ids_list, dtype=torch.long)
        labels_batch = input_ids_batch.clone()

        # 3) MLMマスク処理（mlm=True のときだけ）
        if self.mlm:
            probability_matrix = torch.full(labels_batch.shape, self.mlm_probability)
            special_ids = set([
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.unk_token_id,
                self.tokenizer.pad_token_id
            ])
            mask_arr = torch.bernoulli(probability_matrix).bool()
            for s_id in special_ids:
                mask_arr &= (input_ids_batch != s_id)
            
            input_ids_batch[mask_arr] = self.mask_token_id
            labels_batch[~mask_arr] = -100

        return {
            "input_ids": input_ids_batch,
            "labels": labels_batch
        }