"""
日常のお金の計算(単価×個数、2品の合計、お釣り)を確実に正しく答えるための
ルールベースの電卓機能。

2026-09-09の評価で、モデルに計算を学習データとして覚えさせても
(generated_arithmetic_ja.txt)、学習データに無い数値の組み合わせでは
全く汎化しないことが分かった(見た目の正解は丸暗記だった)。小規模な
Transformerに数字の計算を「理解」させるのは筋が悪いので、そもそも
モデルに解かせず、質問文からルールベースで数値を取り出してPythonの
四則演算で確実に計算する方式に切り替える(実際のLLM製品がやっている
「電卓ツールを呼び出す」のミニ版)。

ここで拾えなかった質問は None を返すので、呼び出し側(chat.py)は
これまで通りモデルに生成させる。

2026-09-10: 並列テストエージェントが実際の誤答パターンを多数発見した
(training_log.md参照)。特に「複数商品購入時のお釣りが中間の商品を
無視する」「日付や電話番号を減算と誤読する」「小数価格が末尾1桁に化ける」
は、静かに確信を持って間違った数値を返す点で、電卓機能の存在意義
(ハルシネーション防止)そのものを損なう最重要バグとして修正した。
"""

from __future__ import annotations

import re

_NUM_UNIT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(円|個|本|枚|つ|冊|杯|匹|台|箱)")
_MAN_EN = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*万\s*円")

# 2026-09-10: 素朴な `\d+\s*-\s*\d+` は日付(2026-09-10)・電話番号(090-1234-5678)・
# 郵便番号(163-8001)・時間帯(10-11時)・話数(第4-5話)まで減算と誤読してしまう。
# 「-」記号による引き算だけは、これらの形に一致しないことを確認してから使う
_DATE_LIKE = re.compile(
    r"\d{4}-\d{1,2}-\d{1,2}|\d{2,4}-\d{3,4}-\d{3,4}|\d{3}-\d{4}|"
    r"\d{1,2}-\d{1,2}\s*(?:時|話)|\d{1,2}-\d{1,2}\s*日"
)
_GENERIC_EXPR = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(\+|足す|プラス|×|x|X|\*|掛ける|かける|-|引く|マイナス|÷|/|割る)\s*(\d[\d,]*(?:\.\d+)?)"
)
_NEGATION_NEARBY = re.compile(r"(聞いてない|関係ない|の話じゃない|じゃなくて|は要らない|はいらない)")


def _to_num(s: str) -> float:
    n = float(s.replace(",", ""))
    return int(n) if n == int(n) else n


def _fmt(n: float) -> str:
    if isinstance(n, float) and n != int(n):
        return f"{n:,.1f}"
    return f"{int(n):,}"


def _extract_price_and_count(text: str) -> tuple[list[float], int | None]:
    """テキスト中の「数字+円」列と「個数」を取り出す。

    「1つ130円で」のような単価表現では、個数系の単位(つ)の直後に
    「円」が続く。これは購入する個数ではなく「1つあたりの価格」の
    言い方なので、個数の候補から除外する。
    2026-09-10: ただし「3000円のシャツを2枚買って、5000円払いました」の
    ように、個数の直後の円が実は離れた「支払い額」で、単価表現ではない
    場合がある。両者の間が助詞程度(数文字)で直結しているときだけ
    単価表現とみなし、動詞(買って等)を挟んで離れている場合は個数として
    扱う
    """
    matches = list(_NUM_UNIT.finditer(text))
    yen_values: list[float] = []
    count_value: int | None = None
    for i, m in enumerate(matches):
        n = _to_num(m.group(1))
        unit = m.group(2)
        if unit == "円":
            yen_values.append(n)
            continue
        is_rate_qualifier = False
        if i + 1 < len(matches) and matches[i + 1].group(2) == "円":
            gap = text[m.end() : matches[i + 1].start()]
            is_rate_qualifier = len(gap) <= 2
        if not is_rate_qualifier and count_value is None:
            count_value = int(n)
    for man_str in _MAN_EN.findall(text):
        yen_values.append(_to_num(man_str) * 10000)
    return yen_values, count_value


def try_calculate(text: str) -> str | None:
    """買い物の計算質問なら、確実な答えを文字列で返す。該当しなければNone。"""
    wants_change = "お釣り" in text or "おつり" in text
    # 2026-09-10: 「いくら」は「イクラ(魚卵)」と同音異義語。「いくらの」
    # 「いくらが」「いくらを」のような名詞用法(直後に助詞)は金額の質問
    # ではないので除外する
    has_price_question = "いくら" in text and not re.search(r"いくら[のがを]", text)
    wants_total = ("合計" in text) or has_price_question
    if wants_total or wants_change:
        # 2026-09-10: 「合計は聞いてないよ」のような否定文では計算しない
        for keyword in ("合計", "いくら"):
            idx = text.find(keyword)
            if idx != -1:
                window = text[max(0, idx - 15) : idx + 15]
                if _NEGATION_NEARBY.search(window):
                    return _try_generic_expr(text)

    if not (wants_change or wants_total):
        return _try_generic_expr(text)

    yen_values, count_value = _extract_price_and_count(text)

    if wants_change and len(yen_values) >= 2:
        # 2026-09-10: 支払い額は語順に関わらず「一番大きい金額」とみなす
        # (「1000円出して300円の物を買うと」のような逆順表現にも対応)。
        # 複数商品(3つ以上のyen_values)の場合、支払い額以外の**全部**を
        # 合計するのが正しい(以前は最初と最後だけを見て中間商品を無視する
        # バグがあった)
        pay = max(yen_values)
        item_prices = [v for v in yen_values if v != pay] or yen_values[:-1]
        if count_value is not None and len(item_prices) == 1:
            price = item_prices[0] * count_value
        else:
            price = sum(item_prices)
        if pay > price:
            change = pay - price
            return (
                f"{_fmt(pay)}円 - {_fmt(price)}円 = {_fmt(change)}円です。"
                f"お釣りは{_fmt(change)}円になります。"
            )
        if pay < price:
            shortage = price - pay
            return (
                f"{_fmt(pay)}円では{_fmt(price)}円に対して{_fmt(shortage)}円"
                f"足りません。お釣りではなく、追加で{_fmt(shortage)}円必要です。"
            )
        return None

    if wants_total and not wants_change:
        if count_value is not None and len(yen_values) == 1:
            price = yen_values[0]
            total = price * count_value
            return (
                f"{_fmt(price)}円 × {count_value} = {_fmt(total)}円です。"
                f"合計{_fmt(total)}円になります。"
            )
        if len(yen_values) == 2 and count_value is None:
            total = yen_values[0] + yen_values[1]
            return (
                f"{_fmt(yen_values[0])}円 + {_fmt(yen_values[1])}円 = "
                f"{_fmt(total)}円です。合計{_fmt(total)}円になります。"
            )

    return _try_generic_expr(text)


def _try_generic_expr(text: str) -> str | None:
    """「7×8は?」のような単純な数式を直接計算する(電卓の最終フォールバック)"""
    m = _GENERIC_EXPR.search(text)
    if not m:
        return None
    op = m.group(2)
    # 2026-09-10: 「-」記号の減算だけは、日付・電話番号・郵便番号・時間帯・
    # 話数などの表記と衝突しやすいため、そのような形に一致する場合は
    # 計算とみなさない(try_datetime等、他のツールに任せる)
    if op == "-" and _DATE_LIKE.search(text):
        return None
    a = _to_num(m.group(1))
    b = _to_num(m.group(3))

    if op in ("+", "足す", "プラス"):
        return f"{_fmt(a)} + {_fmt(b)} = {_fmt(a + b)}です。"
    if op in ("×", "x", "X", "*", "掛ける", "かける"):
        return f"{_fmt(a)} × {_fmt(b)} = {_fmt(a * b)}です。"
    if op in ("-", "引く", "マイナス"):
        return f"{_fmt(a)} - {_fmt(b)} = {_fmt(a - b)}です。"
    if op in ("÷", "/", "割る"):
        if b == 0:
            return None
        if a % b == 0:
            return f"{_fmt(a)} ÷ {_fmt(b)} = {_fmt(a // b)}です。"
        return f"{_fmt(a)} ÷ {_fmt(b)} は約{a / b:.2f}です。"
    return None
