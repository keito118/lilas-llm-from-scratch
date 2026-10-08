# Lilas

[English](README.md) | **日本語**

Lilas（リラ）は、PyTorchでゼロから作った小さな日本語会話モデル（GPT系、40,241,152パラメータ）です。事前学習済みモデルは使っていません。トークナイザー、モデル、学習ループ、ファインチューニング用データ、評価セットはすべてこのリポジトリに入っています。Lilasは秘書・アシスタント風の丁寧な話し方（です/ます調）をします。

固定の評価用会話セットでの実際の出力（`eval/results_round2_2026-09-10.md`、現行モデル、無編集）:

```
User: やあ、久しぶり
Lilas: お久しぶりです!最近どうされていましたか?
User: 元気にしてた?
Lilas: はい、元気にしております。あなたはいかがですか?
```

```
User: ところで130円のジュースを3本買ったら合計いくら?
Lilas: 130円 × 3 = 390円です。合計390円になります。以上になります。
```

同じファイルには、うまく答えられなかった会話（ありきたりな返答、話題の呼び戻しの失敗など）もそのまま残しています。詳しくは[評価結果](#評価結果)を参照してください。

## 特徴

- **ゼロから実装**: 独自のBPEトークナイザー（影響を受けた単語だけを差分更新する方式で学習を28倍高速化）とデコーダ型Transformer。生成時はtop-pサンプリングと、直前のターンにも効く繰り返しペナルティを使用。
- **2段階の学習**: 青空文庫とWikipedia日本語版の一部で事前学習し、dolly-15k-ja、alpaca-gpt4-japanese、自作の会話データでファインチューニング。自作データで秘書風の人格を持たせています。
- **コンテキスト拡張**: 初代モデル（v1）は、学習済みの位置埋め込みをコピーして継続学習する方式で、ゼロから学習し直さずにコンテキスト長を256から1024トークンへ拡張しました（`scripts/extend_context.py`）。現行モデル（v2）は最初から1024トークンで事前学習しています。
- **ツール利用**: ルールベースの電卓、祝日、天気（気象庁）、Wikipedia検索、2つの対象の比較。すべてAPIキー不要の公開APIです。ツールの結果は「(参考情報: ...)」としてモデルに渡し、回答の文章はモデル自身が書きます（RAG方式）。
- **改善を記録で確認**: 固定した評価用会話セット（`eval/held_out_conversations.md` の12会話と、v2セットの6会話）で、学習ラウンドごとに評価。採用・不採用に関わらず結果を `eval/` に残しています。
- **正直な記録**: うまくいかなかった点、原因の調べ方、判断の理由を `training_log.md` と `SCALE_UP_PLAN.md` に記録しています。

## 構成

| 項目 | v1（2026-09-07〜08） | v2・現行（2026-09-09〜） |
|---|---|---|
| モデル | デコーダ型Transformer | デコーダ型Transformer |
| パラメータ数 | 6,852,608 | 40,241,152 |
| 層 / ヘッド / 埋め込み次元 | 6 / 8 / 256 | 10 / 8 / 512 |
| 語彙数（コーパスから学習したBPE） | 8,000 | 16,000 |
| コンテキスト長 | 256 → 1024（拡張） | 1024 |
| 事前学習コーパス | 178,298,814トークン | 青空文庫 + Wikipedia日本語版の一部 |

```
コーパス（青空文庫 + Wikipedia日本語版の一部）
        │  事前学習 (lilas/train.py)
        ▼
   ベースモデル ──► （v1のみ）コンテキスト拡張 256→1024 (scripts/extend_context.py)
        │  指示データ + 人格データでファインチューニング (--curated-only)
        ▼
   会話モデル ──► lilas/chat.py
        │           ├─ lilas/calculator.py（計算）
        │           └─ lilas/tools.py（祝日・天気・Wikipedia・比較）
        ▼
   eval/run_eval.py（固定の評価用会話セット）
```

## フォルダ構成

```
Project_Lilas/
├── README.md / README.ja.md   このファイル（英語版 / 日本語版）
├── training_log.md            学習・実験で実際に躓いたことのトラブルシューティング記録
├── SCALE_UP_PLAN.md           v2へのスケールアップ計画と、ラウンドごとの結果
├── skills.md                  Lilasに必要な知識・スキルのカタログ（人間向け、学習データには含めない）
├── requirements.txt           Python依存パッケージ
├── lilas/                     実装コード
│   ├── config.py              モデル・学習のデフォルト設定
│   ├── tokenizer.py           自作BPEトークナイザ（大規模コーパス向けに高速化）
│   ├── model.py               自作Transformer（top-p・repetition_penalty対応）
│   ├── data.py                学習データの読み込み・バッチ生成（persona_repeat等の複製倍率）
│   ├── train.py               学習スクリプト（事前学習・ファインチューニング両対応）
│   ├── chat.py                チャットで会話するスクリプト（ツールの振り分け）
│   ├── calculator.py          ルールベースの電卓（単価×個数・合計・お釣り・数式）
│   └── tools.py               祝日・気象庁天気・Wikipedia検索・比較ツール
├── scripts/
│   ├── fetch_datasets.py              公開コーパス・instructデータをindex/に取得
│   ├── extend_context.py              チェックポイントのblock_size（文脈長）を拡張
│   ├── generate_seed_data.py          テンプレートから会話データを合成生成
│   ├── generate_greeting_data.py      挨拶・別れ際・相槌・お礼/謝罪の会話
│   ├── generate_emotion_data.py       感情語の無い発言・曖昧な入力・長い会話の質問フォーカス
│   ├── generate_grounded_data.py      「参考情報」付きの質問に答える会話（RAG方式）
│   ├── generate_knowledge_broad.py    実際のWikipediaテキストを使った知識グラウンディング
│   ├── generate_longcontext_data.py   複数ターン会話（数ターン前の話題を呼び戻す形式）
│   └── generate_arithmetic_data.py    計算問題（現在は電卓機能に置き換え済み）
├── eval/
│   ├── run_eval.py                    凍結テストセットをチェックポイントに流すスクリプト
│   ├── held_out_conversations*.md     凍結テストセット（学習には絶対に使わない）
│   ├── results_*.md                   各ラウンドの評価結果
│   └── parallel_test_*.md             カテゴリ別のストレステスト報告
├── index/                     Lilasの学習データ置き場（詳細は index/README.md）
│   ├── persona_lilas.md, about_lilas_ja.txt         人格設定・自己紹介
│   ├── seed_conversations_ja.txt                    手書きの会話サンプル
│   ├── generated_*_ja.txt                           scripts/generate_*.py で生成した会話
│   ├── claude_authored_*_ja.txt                     Claudeが執筆した会話・知識Q&A
│   ├── instruct_*.txt                               （同梱なし）公開instruction tuningデータ
│   └── external_*.txt                               （同梱なし）事前学習用の公開コーパス
└── checkpoints/               （同梱なし）tokenizer.json と学習済みモデル（*.pt）の保存先
```

## 使い方

Python 3.11で動作確認しています。NVIDIAのGPUがある場合は、先にCUDA対応版のPyTorchを入れておくと大幅に速くなります（例: `pip install torch --index-url https://download.pytorch.org/whl/cu121`）。`torch.cuda.is_available()` がTrueなら、train.py / chat.py は自動でGPUを使います。

```bash
pip install -r requirements.txt

# 1. 外部データセットをindex/に取得（リポジトリには含まれていません）
python scripts/fetch_datasets.py all        # 青空文庫 + Wikipediaサンプル（事前学習用）
python scripts/fetch_datasets.py instruct   # dolly-15k-ja + alpaca-gpt4-japanese（ファインチューニング用）

# 2. 事前学習（v2の設定）
python -m lilas.train --vocab-size 16000 --n-embd 512 --n-layer 10 --n-head 8 \
  --block-size 1024 --batch-size 8 --steps 80000 --tag v2

# 3. （任意）既存チェックポイントのコンテキストを拡張し、継続学習で適応させる
python scripts/extend_context.py checkpoints/lilas_pretrained_base.pt \
  checkpoints/lilas_extended_base.pt --new-block-size 1024
python -m lilas.train --init-from checkpoints/lilas_extended_base.pt \
  --steps 15000 --batch-size 16 --lr 0.00015 --tag context_adapted

# 4. 会話用にファインチューニング
python -m lilas.train --init-from checkpoints/lilas_v2.pt --curated-only \
  --persona-repeat 50 --steps 8000 --batch-size 8 --lr 0.0001 --tag v2_ft

# 5. 会話する（デフォルトは checkpoints/lilas.pt。--tag NAME で checkpoints/lilas_NAME.pt）
python -m lilas.chat
python -m lilas.chat --once "こんにちは"

# 6. 凍結テストセットで評価
python eval/run_eval.py --tag v2_ft --out eval/results_v2_ft.md
```

- 初回の事前学習では、トークナイザの学習とコーパスのエンコードに時間がかかります（2回目以降はキャッシュされます）。
- 拡張後は必ず新しいblock_sizeで継続学習（適応）してください。拡張しただけでは、モデルは延長した位置の使い方を知りません。
- 現行モデル（`round2`）は、`lilas_v2.pt` から `--persona-repeat 21`、15,000ステップでファインチューニングしたものです。`chat.py` は `checkpoints/lilas.pt` を読むので、使いたいチェックポイントをこの名前にしてください。
- `chat.py` は `Ctrl+C` か `exit` で終了します。`--top-p` / `--repetition-penalty` でサンプリングを調整できます（デフォルトは config.py の top_p=0.9、repetition_penalty=1.15）。

ファインチューニングの主なオプション:

- `--init-from`: 初期値にするチェックポイント
- `--curated-only`: `index/` の `external_*` ファイルを除外し、手作り+instructデータだけを使う
- `--persona-repeat N`: Lilas固有の人格・会話データをN回複製してコーパスに混ぜる。instructデータ（数千万文字）に対して人格データ（数万文字）が薄すぎると人格が学習されないため、比率を意図的に上げる。コーパス全体に対する比率が15〜20%程度になるよう調整するのが目安（詳細は training_log.md）
- `--tag <name>`: 保存先を `lilas_<name>.pt` にする（ベースモデルを上書きしたくない場合）

**学習済みモデルと外部データセットは、サイズとライセンスの都合で含めていません。** データセットは `scripts/fetch_datasets.py` で取得できます。
<!-- TODO: Hugging Face や GitHub Release で重みを公開する場合はここにリンク -->

## 評価結果

評価セットの結果ファイルはスコアではなく会話そのものなので、ラウンドごとに全会話を読み、現行モデルと比較して判断しました。下の数字は `SCALE_UP_PLAN.md` と `training_log.md` に記録されているものです。

| ラウンド | 変更点 | 評価結果 | 採用 |
|---|---|---|---|
| baseline（v1） | 685万パラメータ、256→1024に拡張 | 単発質問テストは17問中16問だが、複数ターンでは破綻（同じ誤答を3連発） | — |
| v2_ft | 4,024万パラメータに拡大、最初から1024で事前学習 | 文法の破綻がほぼ消えた。富士山の標高（3,776m）に正答 | 採用 |
| v2_ft repfix | 繰り返しペナルティを直前のターンにも適用（デコードのみ） | 会話2・4・6・7のターン間の繰り返しが解消 | 採用 |
| v2_ft2 | 手書きの複数ターン会話8件を追加し2回目のファインチューニング | val lossは同水準（2.7927）だが非文法的な文が新たに出現 | 不採用 |
| v2_ft3 | テンプレート生成のcallback・計算データ、クリーンなv2から学習 | 未知の固有名詞で初めてcallbackが汎化。計算は丸暗記（130×3は正解、90×7は誤答） | 不採用 |
| 電卓 | 計算を学習させず、ルールベースの電卓に置き換え | 会話11の計算問題に確実に正答 | 採用 |
| v2_ft4 | テンプレートの語句をランダムに言い換え | callbackは再現。無関係な場面へのテンプレート漏れ | 不採用 |
| v2_ft5 | 語句ではなく文構造ごとのバリエーションを用意 | テンプレート漏れが解消、callback 3回連続成功。富士山で退行 | 不採用 |
| v2_ft6 | 24トピックに拡大、知識データを3倍に複製 | callback 4回連続成功。富士山は直らず → 知識は外部ツールへ | 不採用 |
| v2_ft8 | ツール＋参考情報（RAG方式）の学習データ714例 | 電卓・祝日・時刻はほぼ確実。callback 7回連続成功 | 採用 |
| v2_ft9 | 参考情報を実際のWikipediaの返り値に近い形に | 東京タワー・富士山・織田信長・琵琶湖が安定。callback 8回連続成功 | 採用 |
| round1 | 挨拶・長いcallback・感情・一貫性データを追加 | 挨拶は改善。未知の人物を別人の事実で答える問題が発生 | 不採用 |
| **round2** | 一貫性データを削除、長い会話の質問フォーカス例を増量 | 一連の実験で最もクリーン。挨拶は16/16で適切 | **採用（現行）** |
| round3 | 長い会話フォーカス例を97件に、比較ツール | 長い会話フォーカス2/2で改善。比較の結論を2/2で取り違え、エベレストで退行 | 不採用 |
| round4 | 比較の対象を単独知識と重ならないものに | エベレストは解消。フォーカス改善が消え、人物混同が再発 | 不採用 |
| round5 | round4と同じデータでシードのみ変更 | フォーカス改善が再現せず → round3の改善はシードの巡り合わせ | 不採用 |
| round6 | 実際のWikipediaテキストで70エンティティを学習 | 学習済みの2人物の名前が融合した架空の人物名を生成 | 不採用 |
| round7 | 70エンティティのみに一本化 | エンティティの融合が形を変えて再発 | 不採用 |
| round8 | 暗示的な感情の学習例を拡充 | 近い言い換えにも汎化せず | 不採用 |

現行モデルでのその他の測定: 感情推論はストレステストで30〜40%から66.7%に改善。知識ツールが発火した質問の正答率は24問中8問（33%）でした。

## 学んだこと

- **val lossは会話の質ではない。** val lossが同じか良いのに、読むと明らかに悪化しているラウンドが何度もありました。val lossが最良のステップより最終ステップの方が自然なこともありました。今は毎ラウンド、凍結テストセットを読んで判断しています。
- **「改善」は学習データに無い入力で確かめる。** 130円×3は解けたように見えましたが、90円×7は「300円」でした。答えは丸暗記だったので計算はルールベースの電卓に移し、のちに知識・比較もツール側で答えを確定させる設計にしました。
- **アルゴリズムの計算量は規模を上げて初めて表に出る。** 最初のBPE学習はマージのたびにコーパス全体を走査していました。影響を受けた単語だけを差分更新する方式にして28倍速くなりました（同じテストで356秒→12.65秒）。
- **推論時の工夫は、モデルを変えるたびに再検証する。** 256トークンでは無害だった「会話の先頭に自己紹介を仕込む」仕組みが、1024トークンのモデルでは無関係な質問にまで自己紹介で答えさせるようになったため、廃止しました。
- **構造的な限界もある。** トークナイザが数字を不規則に分割する（`50000`→`500`+`00`、`45000`→`45`+`000`）ため、文脈中の任意の数字を呼び戻すことは学習できませんでした。約4,000万パラメータでは学習済みのエンティティ同士が混ざる問題も残りました。どちらも、データを増やすのはやめて、答えはコード側で確定させる判断をしました。

## 注意点（推論時の工夫について）

推論時のプロンプト操作を追加するときは、モデルやデータを変更するたびに必ず再検証してください（詳細は training_log.md）。

## データとライセンス

| データ | ライセンス | 同梱 |
|---|---|---|
| 青空文庫（`globis-university/aozorabunko-clean`） | パブリックドメイン（作品ごと） | なし（スクリプトで取得） |
| Wikipedia日本語版の一部（`wikimedia/wikipedia`、20231101.ja） | CC BY-SA | なし（スクリプトで取得） |
| dolly-15k-ja（`kunishou/databricks-dolly-15k-ja`） | CC BY-SA 3.0 | なし（スクリプトで取得） |
| alpaca-gpt4-japanese（`FreedomIntelligence/alpaca-gpt4-japanese`） | CC BY-NC 4.0（GPT-4生成） | なし（スクリプトで取得） |
| `index/` の自作データ | 本リポジトリと同じ | あり |

## ライセンス

MIT License（[LICENSE](LICENSE) を参照）。データセットはそれぞれのライセンスに従います（上の表を参照）。
