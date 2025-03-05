# 現在入手可能な優秀な日本語トークナイザー

日本語処理に最適化されたトークナイザーは、大規模言語モデル（LLM）の性能を左右する重要な要素です。検索結果から得られる情報をもとに、現在入手可能な優秀な日本語トークナイザーについて解説します。

## 主要な日本語トークナイザーとその性能

現在、複数の日本語対応トークナイザーが公開されています。検索結果によると、日本語テキストの処理効率を示す指標「length per token (lpt)」で比較した場合、以下のようなランキングとなっています：

1. cyberagent/open-calm-7b（lpt: 2.05）
2. stockmark/gpt-neox-japanese-1.4b（lpt: 1.99）
3. rinna/bilingual-gpt-neox-4b（lpt: 1.95）
4. NovelAI/nerdstash-tokenizer-v2（lpt: 1.76）
5. google/gemma-2b（lpt: 1.59）
6. llm-jp/llm-jp-13b-v2.0（lpt: 1.51）

このランキングから、日本語に特化したモデルのトークナイザーが日本語処理においてより効率的であることがわかります。lptが高いほど、少ないトークン数で多くの日本語テキストを表現できるため、トークン効率が良いと言えます[3]。

## LLM-jp Tokenizer

特に注目すべきは、LLM-jp Tokenizerです。このトークナイザーはGitHubで公開されており、以下の特徴を持っています：

- SentencePiece（Unigramモード）をベースにしている
- 既存の大規模言語モデル（Mistral）を日本語・中国語・韓国語を対象に拡張した語彙を持つ
- 言語ごとに含めるべき「必須語彙」を指定した段階的な語彙の拡張機能
- 多言語設定に拡張しやすいスケーラブルな学習枠組み

このトークナイザーは以下のGitHubリポジトリから入手可能です：
https://github.com/llm-jp/llm-jp-tokenizer[4]

## その他注目すべきトークナイザー

Japanese StableLM Alphaでは、NovelAIのnovelai-tokenizer-v1（現在はv2も存在）が使用されており、日本語と英語を効率的かつ効果的に処理するための工夫が施されています[1]。NovelAI/nerdstash-tokenizer-v2は日本語と英語の両方で比較的高いlptを示しており、バランスの取れたトークナイザーと言えるでしょう[3]。

また、ELYZA-japanese-Llama-2-7bでは、追加の事前学習においてトークナイザーの改良を行ったことが報告されています[1]。このようなモデル固有のトークナイザーも、特定の用途においては優れた性能を発揮する可能性があります。

## まとめ

「最も優秀」という評価は用途によって異なりますが、現時点で入手可能な優秀な日本語トークナイザーの選択肢として、以下が挙げられます：

1. GitHub上のLLM-jp Tokenizer - 多言語対応と拡張性を重視する場合
2. cyberagent/open-calm-7bのトークナイザー - 純粋な日本語処理効率を重視する場合
3. NovelAI/nerdstash-tokenizer-v2 - 日本語と英語のバランスを重視する場合

これらのトークナイザーは、それぞれのGitHubリポジトリやHugging Faceから入手できます。用途や求める性能に応じて、適切なトークナイザーを選択することが重要です。

Citations:
[1] https://dalab.jp/archives/journal/japanese-llm-tokenizer/
[2] https://www.rondhuit.com/solr%E3%81%AE%E6%97%A5%E6%9C%AC%E8%AA%9E%E5%AF%BE%E5%BF%9C.html
[3] https://zenn.dev/if001/articles/c1f2ae0e203bad
[4] https://github.com/llm-jp/llm-jp-tokenizer
[5] https://soroban.highreso.jp/article/article-058
[6] https://emaillab.jp/spamassassin/ja-patch/
[7] https://www.studio-umi.jp/blog/811
[8] https://zenn.dev/if001/articles/87bbe893411fa1
[9] https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/Q6-1.pdf
[10] https://qiita.com/engchina/items/968cae7abce948cf8f22
[11] https://gigazine.net/news/20230614-llama-tokenizer/
[12] https://zenn.dev/matsuolab/articles/bbd855bf933030
[13] https://www.anlp.jp/proceedings/annual_meeting/2021/pdf_dir/P4-12.pdf
[14] https://www.atilika.com/ja/kuromoji/
[15] https://weel.co.jp/media/tech/swallow/
[16] https://tech.preferred.jp/ja/blog/plamo-2-tokenizer/
[17] https://cn.teldevice.co.jp/blog/p49546/
[18] https://qiita.com/e10persona/items/fddc795e70a05f3bc907
[19] https://megagon.ai/jp/projects/ginza-install-a-japanese-nlp-library-in-one-step/
[20] https://www.paloaltoinsight.com/2022/01/05/tokenizer-nlp-byt5/

---
Perplexity の Eliot より: pplx.ai/share