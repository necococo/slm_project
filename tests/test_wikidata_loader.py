# test_wikidata_loader.py
from wikidata_loader import run_wikiextractor

def test_run_wikiextractor_function():
    # 実際のWikipediaダンプは使わず、関数が呼び出し可能であることだけ確認
    assert callable(run_wikiextractor), "run_wikiextractor should be callable"