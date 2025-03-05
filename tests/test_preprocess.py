# test_preprocess.py
from preprocess import mecab_tokenize

def test_mecab_tokenize():
    text = "これはテストです。"
    tokens = mecab_tokenize(text)
    # トークン列がリストとして返っていること、かつ十分な数のトークンがあることを確認
    assert isinstance(tokens, list), "mecab_tokenize should return a list"
    assert len(tokens) >= 3, "Tokenized result should contain at least 3 tokens"