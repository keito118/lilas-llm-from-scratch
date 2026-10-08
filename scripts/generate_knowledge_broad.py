"""
知識グラウンディングの「丸暗記」問題を根本から解決するためのスクリプト。

2026-09-10、並列テストエージェントが根本原因を特定した: 27個の固定
エンティティ(FACTS)を学習データとして使うと、モデルは「実際に渡された
参考情報を読む」という汎用スキルではなく、「この質問の形にはこの
エンティティの暗記した答え」という近道(ルックアップテーブル)を学習
してしまう。未知のエンティティ(阿蘇山等)を聞かれても、学習済みの
27個のうちどれかの答え(富士山の3,776m等)を差し込んでしまう。

対策: 手書きの27件ではなく、100件以上の多様なエンティティについて、
**実際にWikipedia APIから取得した本物のテキスト**を使って学習データを
作る。事実の暗記先が100件以上に広く薄く分散すれば、「質問の形→
特定の1つの答え」という近道が成立しにくくなり、「参考情報を読んで
答える」という汎用スキルの学習を後押しできる(callbackトピックを
8→32種類に増やしたときと同じ発想)。

このスクリプトは実際にネットワークアクセスしてWikipedia APIを叩くため、
実行に数分かかる。

使い方:
    python scripts/generate_knowledge_broad.py
    -> index/generated_knowledge_broad_ja.txt を書き出す
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lilas.tools import _wikipedia_fact  # noqa: E402

random.seed(37)

OUT_PATH = PROJECT_ROOT / "index" / "generated_knowledge_broad_ja.txt"

import re  # noqa: E402

_HEIGHT_PAT = re.compile(r"[\d,]+(?:\.\d+)?\s*(?:m|メートル)")
_POP_PAT = re.compile(r"人口[^\d]{0,12}[\d,]{3,}\s*人")
_AREA_PAT = re.compile(r"[\d,]+(?:\.\d+)?\s*(?:km2|km²|平方キロメートル)")
_DIST_PAT = re.compile(r"[\d,]+(?:\.\d+)?\s*km(?![²2])|[\d,]+(?:\.\d+)?\s*キロメートル")

# (エンティティ名, 種類, 検出パターン, 質問テンプレート群)
ENTITIES: list[tuple[str, str, "re.Pattern | None", list[str]]] = [
    # 山(height)
    ("阿蘇山", "height", _HEIGHT_PAT, ["{topic}の高さってどれくらいだっけ?", "{topic}の高さは?"]),
    ("槍ヶ岳", "height", _HEIGHT_PAT, ["{topic}の高さは?", "{topic}ってどのくらい高いの?"]),
    ("穂高岳", "height", _HEIGHT_PAT, ["{topic}の高さってどれくらい?"]),
    ("大雪山", "height", _HEIGHT_PAT, ["{topic}の高さは?"]),
    ("羊蹄山", "height", _HEIGHT_PAT, ["{topic}の高さってどれくらいだっけ?"]),
    ("キリマンジャロ", "height", _HEIGHT_PAT, ["{topic}の高さは?", "{topic}ってどのくらい高いの?"]),
    ("モンブラン", "height", _HEIGHT_PAT, ["{topic}の高さってどれくらい?"]),
    ("マッターホルン", "height", _HEIGHT_PAT, ["{topic}の高さは?"]),
    ("K2", "height", _HEIGHT_PAT, ["{topic}の高さってどれくらいだっけ?"]),
    ("マウナケア", "height", _HEIGHT_PAT, ["{topic}の高さは?"]),
    # 建物・タワー(height)
    ("大阪城", "height", _HEIGHT_PAT, ["{topic}の高さは?"]),
    ("名古屋城", "height", _HEIGHT_PAT, ["{topic}の高さってどれくらい?"]),
    ("横浜ランドマークタワー", "height", _HEIGHT_PAT, ["{topic}の高さは?"]),
    ("さっぽろテレビ塔", "height", _HEIGHT_PAT, ["{topic}の高さってどれくらいだっけ?"]),
    ("福岡タワー", "height", _HEIGHT_PAT, ["{topic}の高さは?"]),
    ("エッフェル塔", "height", _HEIGHT_PAT, ["{topic}の高さは?", "{topic}ってどのくらい高いの?"]),
    ("自由の女神", "height", _HEIGHT_PAT, ["{topic}の高さってどれくらい?"]),
    ("ビッグベン", "height", _HEIGHT_PAT, ["{topic}の高さは?"]),
    ("ピサの斜塔", "height", _HEIGHT_PAT, ["{topic}の高さってどれくらいだっけ?"]),
    ("バベルの塔", "height", None, ["{topic}って何?"]),
    # 湖・川(area/distance)
    ("諏訪湖", "area", _AREA_PAT, ["{topic}の広さってどれくらい?", "{topic}の面積は?"]),
    ("中禅寺湖", "area", _AREA_PAT, ["{topic}の面積は?"]),
    ("摩周湖", "area", _AREA_PAT, ["{topic}の広さってどれくらい?"]),
    ("十和田湖", "area", _AREA_PAT, ["{topic}の面積は?"]),
    ("浜名湖", "area", _AREA_PAT, ["{topic}の広さってどれくらい?"]),
    ("最上川", "distance", _DIST_PAT, ["{topic}の長さってどれくらい?"]),
    ("天竜川", "distance", _DIST_PAT, ["{topic}の長さは?"]),
    ("木曽川", "distance", _DIST_PAT, ["{topic}の長さってどれくらいだっけ?"]),
    ("アマゾン川", "distance", _DIST_PAT, ["{topic}の長さってどれくらい?"]),
    ("長江", "distance", _DIST_PAT, ["{topic}の長さは?"]),
    # 都道府県・都市(area/population/where)
    ("神奈川県", "area", _AREA_PAT, ["{topic}の面積は?"]),
    ("愛知県", "area", _AREA_PAT, ["{topic}の広さってどれくらい?"]),
    ("福岡県", "area", _AREA_PAT, ["{topic}の面積は?"]),
    ("沖縄県", "area", _AREA_PAT, ["{topic}の広さってどれくらい?"]),
    ("京都市", "where", None, ["{topic}はどこにあるの?", "{topic}ってどこ?"]),
    ("神戸市", "where", None, ["{topic}はどこにあるの?"]),
    ("金沢市", "where", None, ["{topic}ってどこ?"]),
    ("松山市", "where", None, ["{topic}はどこにあるの?"]),
    # 歴史人物(who)
    ("福沢諭吉", "who", None, ["{topic}ってどんな人だっけ?", "{topic}って誰?"]),
    ("伊藤博文", "who", None, ["{topic}ってどんな人だっけ?"]),
    ("西郷隆盛", "who", None, ["{topic}って誰?", "{topic}ってどんな人だったの?"]),
    ("勝海舟", "who", None, ["{topic}ってどんな人だっけ?"]),
    ("平賀源内", "who", None, ["{topic}って誰?"]),
    ("与謝野晶子", "who", None, ["{topic}ってどんな人だっけ?"]),
    ("宮沢賢治", "who", None, ["{topic}って誰?", "{topic}ってどんな人だったの?"]),
    ("太宰治", "who", None, ["{topic}ってどんな人だっけ?"]),
    ("葛飾北斎", "who", None, ["{topic}って誰?"]),
    ("千利休", "who", None, ["{topic}ってどんな人だっけ?"]),
    ("卑弥呼", "who", None, ["{topic}って誰?", "{topic}ってどんな人だったの?"]),
    ("源義経", "who", None, ["{topic}ってどんな人だっけ?"]),
    # 動物・自然(what)
    ("キリン", "what", None, ["{topic}って何?", "{topic}とは?"]),
    ("パンダ", "what", None, ["{topic}って何?"]),
    ("ラッコ", "what", None, ["{topic}とは?"]),
    ("カピバラ", "what", None, ["{topic}って何?"]),
    ("フクロウ", "what", None, ["{topic}とは?"]),
    ("イルカ", "what", None, ["{topic}って何?"]),
    ("サンゴ礁", "what", None, ["{topic}とは?", "{topic}って何?"]),
    ("オーロラ", "what", None, ["{topic}って何?"]),
    # 科学・技術(what)
    ("光合成", "what", None, ["{topic}とは?", "{topic}って何?"]),
    ("DNA", "what", None, ["{topic}って何?"]),
    ("ブラックホール", "what", None, ["{topic}とは?"]),
    ("人工知能", "what", None, ["{topic}って何?"]),
    ("インターネット", "what", None, ["{topic}とは?"]),
    ("再生可能エネルギー", "what", None, ["{topic}って何?"]),
    # 食べ物・文化(what/起源)
    ("天ぷら", "what", None, ["{topic}って何?", "{topic}とは?"]),
    ("お好み焼き", "what", None, ["{topic}とは?"]),
    ("茶道", "what", None, ["{topic}って何?"]),
    ("歌舞伎", "what", None, ["{topic}とは?"]),
    ("俳句", "what", None, ["{topic}って何?"]),
    ("origami", "what", None, ["{topic}って何?"]),
]


def build_answer(topic: str, intro: str, detail: str) -> list[str]:
    text = (intro + " " + detail).strip()
    templates = [
        f"{text}",
        f"調べたところ、{text}",
        f"{topic}についてですね。{text}",
    ]
    return templates


ANSWER_PREFIX_POOL = [
    "はい、", "調べたところ、", "", "{topic}についてですね、", "Wikipediaによると、",
]


def main() -> None:
    blocks: list[str] = []
    ok, fail = 0, 0
    for i, (topic, kind, detail_pattern, q_templates) in enumerate(ENTITIES):
        try:
            fact_result = _wikipedia_fact(topic, detail_pattern)
        except Exception as e:
            print(f"  [{i}] {topic}: EXCEPTION {e}")
            fail += 1
            continue
        if fact_result is None:
            print(f"  [{i}] {topic}: NOT FOUND")
            fail += 1
            continue
        title, intro, detail = fact_result
        short = (intro + " " + detail).strip()[:250]
        fact = f"Wikipediaで調べたところ、{title}について: {short}"

        for q_tmpl in q_templates:
            q = q_tmpl.format(topic=topic)
            for prefix in random.sample(ANSWER_PREFIX_POOL, k=2):
                p = prefix.format(topic=topic)
                a = f"{p}{short}".strip()
                blocks.append(f"User: {q}(参考情報: {fact})\nLilas: {a}")
        ok += 1
        if i % 10 == 0:
            print(f"  progress: {i+1}/{len(ENTITIES)} (ok={ok}, fail={fail})")
        time.sleep(0.05)

    random.shuffle(blocks)
    text = "\n\n".join(blocks) + "\n"
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"書き出し: {OUT_PATH}")
    print(f"  エンティティ数: {ok} (失敗: {fail})")
    print(f"  例文数: {len(blocks)}")
    print(f"  文字数: {len(text):,}")


if __name__ == "__main__":
    main()
