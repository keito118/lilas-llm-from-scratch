# index/ — Lilasの学習データ置き場

ここに置いた `.txt` / `.md` ファイルはすべて `lilas/data.py` に読み込まれ、
学習コーパスの一部になる(このREADME自体は除外される)。

## ルール

- 文字コードはUTF-8
- ファイルは自由に増やしてよい。サブフォルダを作っても再帰的に読み込む
- 会話データは以下の形式で書くと、Lilasがターン交代を学習しやすい:

```
User: こんにちは
Lilas: こんにちは、Lilasだよ。今日はどうしたの?

User: 質問があるんだけど
Lilas: うん、なんでも聞いて。
```

- 1つの会話ブロックの間は空行で区切る
- 会話以外の文章(説明文、エッセイ、設定資料など)もそのまま置いてよい。
  文体や語彙の幅を広げるのに役立つ

## 現在のファイル

- `persona_lilas.md` — Lilasの人格・話し方の設定
- `about_lilas_ja.txt` — Lilas向けの短い自己紹介文(誰が作ったか、何ができないか等)。
  ルートの `skills.md`(人間向けの生の開発ログ)は学習に使わないので、
  Lilasに知っておいてほしいことはここに会話向けの文章として書く
- `seed_conversations_ja.txt` — 手書きの会話サンプル(量はまだ少ない、要追加)
- `generated_conversations_ja.txt` — `scripts/generate_seed_data.py` で自動生成した会話データ
- `claude_authored_conversations_ja.txt` — Claudeが直接執筆した会話例(混同しやすい質問の
  言い換えバリエーション、一般知識、相談・感情サポート系など)。モデルの弱点が分かったときに
  ピンポイントで狙って追記するのに向いている
- `claude_authored_general_ja.txt` — Claude執筆の一般的な雑談(トピックの厚みを増やす用)
- `claude_authored_knowledge_ja.txt` — Claude執筆の一般知識Q&A(地理・科学・計算等)
- `claude_authored_multiturn_ja.txt` — Claude執筆の短めの複数ターン会話
- `claude_authored_longcontext_ja.txt` / `claude_authored_longcontext2_ja.txt` — Claude執筆の
  複数ターン会話。数ターン前に話した内容を「さっき話した〜」のように呼び戻す形式で、
  文脈追跡(block_size拡張後の長い会話履歴を活かす力)を鍛える狙い。2は言い回しと
  話題をさらに広げたもの
- `instruct_dolly_ja.txt` / `instruct_alpacagpt4_ja.txt` — 公開instruction tuningデータ
  (詳細はfetch_datasets.py参照)
- `external_aozora_ja.txt` — 青空文庫(公開コーパス、`scripts/fetch_datasets.py`で取得)
- `external_wikipedia_ja_sample.txt` — Wikipedia日本語サンプル(公開コーパス、同上)

`external_` で始まるファイルは「事前学習用の大規模コーパス」の目印。
`lilas/train.py --curated-only`(ファインチューニング用)ではこの接頭辞の
ファイルは除外される。新しく大規模な公開コーパスを追加するときも、
ファイル名を`external_`で始めておくとこの仕組みに乗る。

`lilas/data.py`の`CORE_PERSONA_FILES`に登録されたファイルは、
`--persona-repeat`で複製してコーパス中の比率を上げられる。新しく
Lilas固有の会話データファイルを追加したら、このセットにも追加すること。

## 今後追加すべきもの

- 複数ターンの文脈追跡を「特定の例の暗記」ではなく「汎化したスキル」にするための、
  さらに多様な複数ターン会話データ(2026-09-08時点でまだ弱い部分)
- もっと多様なトピックの会話サンプル
- 「知らないことを正直に言う」以外のパターン(未知の質問により具体的に踏み込む練習)
