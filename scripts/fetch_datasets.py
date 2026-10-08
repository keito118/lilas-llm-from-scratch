"""
公開されている日本語コーパスをHugging Face `datasets` 経由で取得し、
index/ 以下にテキストファイルとして保存するスクリプト。

ディスク容量(このマシンは空き62GB程度)と、素朴な自作BPEトークナイザの
処理速度の都合上、取得範囲を絞っている:

- 青空文庫(globis-university/aozorabunko-clean): 全体で圧縮約240MBと
  現実的なサイズなので全件取得する。物語・会話文が豊富で日本語の
  文章表現を学ぶのに向いている
- 日本語Wikipedia(wikimedia/wikipedia, 20231101.ja): 全体は数GB規模なので
  streaming=Trueで最初のN記事だけをサンプリングする

mC4/CC-100/LLM-jpのような数百GB規模の巨大Webコーパスは、今のLilasの
モデル規模(100万パラメータ未満)には過大で、ディスク容量的にも
全量は扱えないため、このスクリプトでは対象外にしている
(必要になったらstreaming=Trueで同様に少量サンプリングする形で追加できる)。

使い方:
    python scripts/fetch_datasets.py aozora
    python scripts/fetch_datasets.py wikipedia --max-articles 5000
    python scripts/fetch_datasets.py instruct
    python scripts/fetch_datasets.py all

`instruct`(dolly-15k-ja / alpaca-gpt4-japanese)は「User: .../Lilas: ...」の
対話フォーマットに変換して保存する。ファイル名を`external_`で始めていないので
`lilas/train.py --curated-only`(ファインチューニング)の対象になる
(青空文庫やWikipediaのような事前学習用の大規模コーパスとは役割が違うため)。

注意: alpaca-gpt4-japaneseはGPT-4の出力を元にしたデータセット。個人の学習・
研究目的での利用を前提としている。商用利用や再配布を行う場合は元データセットの
ライセンス・利用規約を確認すること。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "index"

AOZORA_OUT = INDEX_DIR / "external_aozora_ja.txt"
WIKIPEDIA_OUT = INDEX_DIR / "external_wikipedia_ja_sample.txt"
DOLLY_OUT = INDEX_DIR / "instruct_dolly_ja.txt"
ALPACA_OUT = INDEX_DIR / "instruct_alpacagpt4_ja.txt"


def fetch_aozora() -> None:
    print("青空文庫(globis-university/aozorabunko-clean)を取得中...")
    ds = load_dataset("globis-university/aozorabunko-clean", split="train")
    print(f"  作品数: {len(ds):,}")

    total_chars = 0
    with open(AOZORA_OUT, "w", encoding="utf-8") as f:
        for row in ds:
            text = (row.get("text") or "").strip()
            if len(text) < 20:
                continue
            f.write(text)
            f.write("\n\n")
            total_chars += len(text)

    print(f"  書き出し: {AOZORA_OUT} ({total_chars:,} 文字)")


def fetch_wikipedia(max_articles: int, max_chars: int) -> None:
    print(f"日本語Wikipediaをストリーミング取得中(最大{max_articles:,}記事)...")
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.ja", split="train", streaming=True
    )

    total_chars = 0
    n_articles = 0
    with open(WIKIPEDIA_OUT, "w", encoding="utf-8") as f:
        for row in ds:
            text = (row.get("text") or "").strip()
            if len(text) < 50:
                continue
            f.write(text)
            f.write("\n\n")
            total_chars += len(text)
            n_articles += 1
            if n_articles >= max_articles or total_chars >= max_chars:
                break

    print(f"  書き出し: {WIKIPEDIA_OUT} ({n_articles:,} 記事, {total_chars:,} 文字)")


def fetch_dolly() -> None:
    print("dolly-15k-ja(kunishou/databricks-dolly-15k-ja)を取得中...")
    ds = load_dataset("kunishou/databricks-dolly-15k-ja", split="train")
    print(f"  件数: {len(ds):,}")

    blocks = []
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        context = (row.get("input") or "").strip()
        output = (row.get("output") or "").strip()
        if not instruction or not output:
            continue
        user_turn = f"{instruction}\n{context}" if context else instruction
        blocks.append(f"User: {user_turn}\nLilas: {output}")

    text = "\n\n".join(blocks) + "\n"
    DOLLY_OUT.write_text(text, encoding="utf-8")
    print(f"  書き出し: {DOLLY_OUT} ({len(blocks):,} 件, {len(text):,} 文字)")


def fetch_alpaca_gpt4(max_rows: int) -> None:
    print(f"alpaca-gpt4-japanese(FreedomIntelligence/alpaca-gpt4-japanese)を取得中(最大{max_rows:,}件)...")
    ds = load_dataset("FreedomIntelligence/alpaca-gpt4-japanese", split="train")
    print(f"  全件数: {len(ds):,}")

    blocks = []
    for row in ds:
        turns = row.get("conversations") or []
        # human/gptが交互に並んでいる想定。ペアごとにUser/Lilasへ変換する
        pending_user = None
        for turn in turns:
            role = turn.get("from")
            value = (turn.get("value") or "").strip()
            if not value:
                continue
            if role == "human":
                pending_user = value
            elif role == "gpt" and pending_user is not None:
                blocks.append(f"User: {pending_user}\nLilas: {value}")
                pending_user = None
        if len(blocks) >= max_rows:
            break

    text = "\n\n".join(blocks) + "\n"
    ALPACA_OUT.write_text(text, encoding="utf-8")
    print(f"  書き出し: {ALPACA_OUT} ({len(blocks):,} 件, {len(text):,} 文字)")


def main() -> None:
    p = argparse.ArgumentParser(description="公開日本語コーパスの取得")
    p.add_argument("target", choices=["aozora", "wikipedia", "instruct", "all"])
    p.add_argument("--max-articles", type=int, default=5000)
    p.add_argument("--max-chars", type=int, default=30_000_000)
    p.add_argument("--max-instruct-rows", type=int, default=50_000)
    args = p.parse_args()

    INDEX_DIR.mkdir(exist_ok=True)

    if args.target in ("aozora", "all"):
        fetch_aozora()
    if args.target in ("wikipedia", "all"):
        fetch_wikipedia(args.max_articles, args.max_chars)
    if args.target in ("instruct", "all"):
        fetch_dolly()
        fetch_alpaca_gpt4(args.max_instruct_rows)


if __name__ == "__main__":
    main()
