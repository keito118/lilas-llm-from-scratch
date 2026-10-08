"""
簡単な日常の計算(単価×個数、合計、お釣り)をテンプレートで大量生成するスクリプト。

2026-09-09の評価で、単純な計算("130円のジュースを3本買ったら合計いくら?")に
Lilasが全く対応できないことが分かった。小規模なTransformerが数字の計算を
本当に「理解」して汎化するのは難しいが、日常会話でよく出る形(単価×個数、
2品の合計、支払いとお釣り)に限定し、計算式を文中に明示しながら大量の
数値バリエーションで学習させることで、少なくともこの種の質問への
「型」だけでも身につくかどうかを試す。

使い方:
    python scripts/generate_arithmetic_data.py
    -> index/generated_arithmetic_ja.txt を書き出す
"""

from __future__ import annotations

import random
from pathlib import Path

random.seed(11)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "index" / "generated_arithmetic_ja.txt"

items = [
    "ジュース", "パン", "おにぎり", "りんご", "みかん", "ノート", "ペン",
    "お菓子", "コーヒー", "卵", "牛乳", "アイス", "チョコレート", "缶詰",
]

item2 = [
    "サンドイッチ", "クッキー", "バナナ", "消しゴム", "紅茶", "せんべい",
]

lead_ins = [
    "{price}円の{item}を{count}本買ったら合計いくら?",
    "{item}が1つ{price}円で、{count}個買うと合計いくらになる?",
    "{item}を{count}個買いたいんだけど、1個{price}円だと全部でいくら?",
    "{price}円の{item}、{count}個で合計はいくらになりますか?",
]

blocks: list[str] = []


def yen(n: int) -> str:
    return f"{n:,}"


# ---- 単価×個数(掛け算) ----
for _ in range(220):
    item = random.choice(items)
    price = random.choice([50, 60, 80, 100, 120, 130, 150, 180, 200, 250, 300, 350, 400])
    count = random.choice([2, 3, 4, 5, 6])
    total = price * count
    q = random.choice(lead_ins).format(price=price, item=item, count=count)
    a = (
        f"{yen(price)}円 × {count} = {yen(total)}円です。"
        f"{item}を{count}本(個)買うと、合計{yen(total)}円になります。"
    )
    blocks.append(f"User: {q}\nLilas: {a}")

# ---- 2品の合計(足し算) ----
for _ in range(150):
    i1, i2 = random.sample(items, 2) if random.random() < 0.5 else (
        random.choice(items), random.choice(item2)
    )
    p1 = random.choice([80, 100, 120, 150, 180, 200, 250, 300])
    p2 = random.choice([80, 100, 120, 150, 180, 200, 250, 300])
    total = p1 + p2
    q = f"{i1}が{yen(p1)}円で、{i2}が{yen(p2)}円のとき、合計いくらになる?"
    a = f"{yen(p1)}円 + {yen(p2)}円 = {yen(total)}円です。合計{yen(total)}円になります。"
    blocks.append(f"User: {q}\nLilas: {a}")

# ---- 支払いとお釣り(引き算) ----
for _ in range(150):
    item = random.choice(items)
    price = random.choice([50, 80, 100, 120, 150, 180, 200, 250, 300, 350, 400, 450])
    pay = random.choice([500, 1000, 2000])
    if pay <= price:
        pay = 1000 + price
    change = pay - price
    q = f"{item}を{yen(price)}円で買って、{yen(pay)}円を出したら、お釣りはいくら?"
    a = f"{yen(pay)}円 - {yen(price)}円 = {yen(change)}円です。お釣りは{yen(change)}円になります。"
    blocks.append(f"User: {q}\nLilas: {a}")

# ---- 雑談を1往復挟んでから聞く、少し長い形も少量混ぜる ----
casual_pairs = [
    ("今日はいい天気だね", "それはいいですね、気持ちのよい一日になりそうですね。"),
    ("さっきお昼を食べてきたよ", "それはよろしいですね、お腹も満たされたのではないでしょうか。"),
    ("週末は買い物に行く予定なんだ", "それは楽しみですね。"),
]
for _ in range(80):
    item = random.choice(items)
    price = random.choice([100, 120, 150, 180, 200, 250, 300])
    count = random.choice([2, 3, 4, 5])
    total = price * count
    cu, cr = random.choice(casual_pairs)
    q = random.choice(lead_ins).format(price=price, item=item, count=count)
    a = (
        f"{yen(price)}円 × {count} = {yen(total)}円です。"
        f"{item}を{count}本(個)買うと、合計{yen(total)}円になります。"
    )
    blocks.append(f"User: {cu}\nLilas: {cr}\nUser: {q}\nLilas: {a}")


def main() -> None:
    random.shuffle(blocks)
    text = "\n\n".join(blocks) + "\n"
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"書き出し: {OUT_PATH}")
    print(f"  例文数: {len(blocks)}")
    print(f"  文字数: {len(text):,}")


if __name__ == "__main__":
    main()
