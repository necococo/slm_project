# Dataset & Tokenizer

```
from datasets import load_dataset
dataset = load_dataset("toramaru-u/wiki40b-ja")

# megagonlabs/t5-base-japanese-web トークナイザーをロード
from transformers import AutoTokenizer
from slm.tokenizer import JapaneseTokenizer # JapaneseTokenizerラッパーに変換するため

jp_tokenizer = JapaneseTokenizer.from_pretrained_tokenizer(hf_tokenizer)

>>>
トークナイザーを保存しました: /content/drive/MyDrive/slm/data/wiki40b_ja/tokenizer
```

トークナイザー情報:megagonlabs/t5-base-japanese-web
元のトークナイザー: T5TokenizerFast
語彙サイズ: 32102
MASK: \<mask> # mask_token_id:32100
PAD: \<pad>
EOS: \</s>
BOS: \<s> # 文の始まりトークンを一応入れた