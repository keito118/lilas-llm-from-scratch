"""
モデルに「覚えさせる」のではなく、必要な時に外部から確実な情報を
取ってきて答えるためのツール群(lilas/calculator.pyの知識版)。

2026-09-09、小規模モデルに個別の事実(富士山の標高等)を学習させても、
学習データの複製倍率をどれだけ上げても簡単に上書き・誤答されることが
分かった(training_log.md参照)。ここでは代わりに、質問文からルール
ベースで「何が聞かれているか」を判定し、無料・APIキー不要の外部API
(Wikipedia日本語版, 気象庁, 祝日API)や、ローカルで確実に分かる情報
(現在時刻)で確実な答えを組み立てる。

いずれの関数も、該当しない・取得に失敗した場合はNoneを返す。
呼び出し側(chat.py)はNoneならこれまで通りモデルに生成させる。
"""

from __future__ import annotations

import datetime as _dt
import re

import requests

_TIMEOUT = 4.0  # 秒。会話のテンポを崩さないよう短めに設定

# WikipediaのAPIはUser-Agent未設定のリクエストを403で拒否するポリシーがある
# (https://w.wiki/4wJS)。プロジェクト名+連絡先相当の情報を名乗る
_WIKI_HEADERS = {
    "User-Agent": "ProjectLilas/1.0 (personal hobby project; https://github.com/)"
}

_WEEKDAY_JA = ["月", "火", "水", "木", "金", "土", "日"]

# 気象庁の発表区(overview_forecast用のコード)。主要都市のみ対応、
# 該当しなければ東京をデフォルトにする
_JMA_OFFICES = {
    "東京": "130000", "横浜": "140000", "神奈川": "140000",
    "大阪": "270000", "京都": "260000", "名古屋": "230000", "愛知": "230000",
    "札幌": "016000", "北海道": "016000", "仙台": "040000", "福岡": "400000",
    "広島": "340000", "沖縄": "471000", "那覇": "471000",
}


def try_datetime(text: str) -> str | None:
    """「今何時?」「今日は何日?」「今日は何曜日?」に、ローカル時刻で答える。

    外部APIを使わないのは、実行しているPC自体が正確な時刻を知っているため
    (ネット越しに調べる必要がそもそも無く、その方が速くて確実)。
    """
    # 2026-09-10: 他のツール(天気・祝日・知識)と違って質問らしさの
    # チェックが無く、「今何時か知らないけど、遅刻した」のような単なる
    # 発言や、「そのドラマ、今何時から?」(番組の開始時刻を聞いている)
    # にまで無関係な現在時刻を割り込ませてしまう不具合があった
    if not _looks_like_question(text):
        return None

    now = _dt.datetime.now()
    weekday = _WEEKDAY_JA[now.weekday()]

    if re.search(r"今.{0,3}何時", text) and not re.search(r"今.{0,3}何時.{0,2}(から|まで|に|の)", text):
        return f"今は{now.hour}時{now.minute}分です。"
    if "何日" in text and ("今日" in text or "きょう" in text):
        return f"今日は{now.year}年{now.month}月{now.day}日です。"
    if ("何曜日" in text or "なんようび" in text) and ("今日" in text or "きょう" in text):
        return f"今日は{weekday}曜日です。"
    return None


def _looks_like_question(text: str) -> bool:
    """「天気いいね」のような発言(文)と、「天気は?」のような質問を区別する。

    2026-09-09: キーワードの有無だけで判定すると、「天気がいいから
    洗濯物が乾きそう」のような単なる発言にも反応してしまい、会話の
    流れを崩す不具合が見つかった。質問らしい語尾かどうかも追加でチェックする。
    """
    return bool(re.search(r"[?？]|だっけ|教えて|どうかな|どう[?？]?$", text))


def try_holiday(text: str) -> str | None:
    """「今日は祝日?」「次の祝日はいつ?」に、祝日APIで確実に答える。"""
    if "祝日" not in text or not _looks_like_question(text):
        return None

    today = _dt.date.today()
    try:
        resp = requests.get(
            "https://holidays-jp.github.io/api/v1/date.json", timeout=_TIMEOUT
        )
        resp.raise_for_status()
        holidays: dict[str, str] = resp.json()
    except Exception:
        return None

    today_str = today.isoformat()
    if today_str in holidays:
        return f"はい、今日({today_str})は「{holidays[today_str]}」です。"

    if "今日" in text or "きょう" in text:
        return f"いいえ、今日({today_str})は祝日ではありません。"

    upcoming = sorted(d for d in holidays if d > today_str)
    if not upcoming:
        return None
    next_date = upcoming[0]
    name = holidays[next_date]
    days_left = (_dt.date.fromisoformat(next_date) - today).days
    return f"次の祝日は{next_date}の「{name}」です(あと{days_left}日)。"


def try_weather(text: str) -> str | None:
    """「今日の天気は?」に、気象庁の発表文で確実に答える。

    JMAの構造化データ(天気コード等)は解析が煩雑なため、既に自然文で
    まとまっている「概況(overview_forecast)」をそのまま使う。
    """
    if "天気" not in text or not _looks_like_question(text):
        return None

    office_code = "130000"
    place = "東京"
    for name, code in _JMA_OFFICES.items():
        if name in text:
            office_code, place = code, name
            break

    try:
        resp = requests.get(
            f"https://www.jma.go.jp/bosai/forecast/data/overview_forecast/{office_code}.json",
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        overview = data.get("text", "").strip()
    except Exception:
        return None

    if not overview:
        return None
    return f"気象庁発表({place}地方)の予報です。\n{overview}"


# (正規表現, 種類, その種類の具体的な数字を探すための正規表現)
# 「単位語を含む」という緩い判定だと、人口を聞いているのに関係ない
# 「〜人には」のような一文を拾ってしまう(2026-09-09発見)ため、
# 「数字+単位」の形をピンポイントで探す。要約の先頭数文に無い場合、
# 記事本文からこのパターンを追加で探す(東京タワー等、要約の冒頭には
# 高さが無く「概要」節に初めて出てくる記事が多かったため)
_KNOWLEDGE_TRIGGERS: list[tuple[re.Pattern, str, re.Pattern | None]] = [
    (re.compile(r"(.+?)の高さ"), "height", re.compile(r"[\d,]+(?:\.\d+)?\s*(?:m|メートル)")),
    (re.compile(r"(.+?)の人口"), "population", re.compile(r"人口[^\d]{0,12}[\d,]{3,}\s*人")),
    (re.compile(r"(.+?)の(?:首都)"), "what", None),
    (re.compile(r"(.+?)の(?:国旗|国歌)"), "what", None),
    (re.compile(r"(.+?)の通貨"), "what", re.compile(r"通貨(?:単位)?は[^\s、。]{1,10}")),
    (re.compile(r"(.+?)の距離"), "what", re.compile(r"[\d,]+(?:\.\d+)?\s*(?:km|キロメートル|m|メートル)")),
    (re.compile(r"(.+?)の値段"), "what", None),
    (re.compile(r"(.+?)の面積"), "area", re.compile(r"[\d,]+(?:\.\d+)?\s*(?:km2|km²|平方キロメートル|ヘクタール|坪)")),
    (re.compile(r"(.+?)の(?:寿命|平均寿命)"), "what", re.compile(r"[\d,]+(?:\.\d+)?\s*年")),
    (re.compile(r"(.+?)(?:を|について)知ってる"), "what", None),
    (re.compile(r"(.+?)の県庁所在地"), "where", None),
    (re.compile(r"(.+?)は何県"), "where", None),
    (re.compile(r"(.+?)はいつ"), "what", None),
    (re.compile(r"(.+?)って何"), "what", None),
    (re.compile(r"(.+?)とは(?:何|なに)?[?？]"), "what", None),
    # 2026-09-10: 「Xは何ですか?」という最も一般的な丁寧語の言い回しを追加。
    # 「とは」より後ろに置き、「とは」でマッチしなかった場合のみ試す
    (re.compile(r"(.+?)は(?:何|なに)です"), "what", None),
    (re.compile(r"(.+?)は誰"), "who", None),
    (re.compile(r"(.+?)って誰"), "who", None),
    # 2026-09-10: eval/held_out_conversations_v2.mdのテストで、
    # 「〜ってどんな人だっけ?」という言い方が「は誰/って誰」の
    # どちらにも一致せず、素通りしてしまうことが分かった
    (re.compile(r"(.+?)(?:は|って)どんな人"), "who", None),
    (re.compile(r"(.+?)はどこ"), "where", re.compile(r"[^\s、。]{2,6}[都道府県]")),
    (re.compile(r"(.+?)の意味"), "what", None),
]


_GREETING_WORDS = ("こんにちは", "こんばんは", "おはよう", "はじめまして", "やあ", "どうも")

# 2026-09-10発見: 「あなたって誰?」のような、Lilas自身に向けられた
# 二人称の質問が、「あなた」という単語そのもののWikipedia記事(代名詞の
# 語源解説)を引っ張ってきてしまう不具合があった。当初は語のリストで
# 除外していたが、「あなた様」(漢字)のような表記違いや「てめえ」
# 「貴殿」等の別語が漏れ続けたため、敬称サフィックスを剥がして
# ベースの代名詞と比較する方式に変更(パターンマッチの方が保守性が高い)
_SELF_REFERENCE_BASE = {
    "あなた", "君", "きみ", "そちら", "お前", "あんた", "てめえ", "貴殿", "貴様",
    # 2026-09-10発見: 「これは何ですか?」「それって何?」のような指示語も、
    # 直前の文脈が無いとWikipedia検索の対象にはなりえない(検索すると
    # 「これ」というタイトルの無関係な作品名等に化けてしまう)ため除外する
    "これ", "それ", "あれ", "こいつ", "そいつ", "あいつ",
}
_HONORIFIC_SUFFIX = re.compile(r"(様|さま|さん)$")
# 2026-09-10発見: 「君の名前は何ですか?」のような、Lilas自身の名前を
# 尋ねる非常に自然な質問が、「君の名前」を丸ごとWikipedia検索してしまい、
# 無関係な楽曲・作品名がヒットして偽の名前や日付を答える重大な不具合が
# あった。二人称代名詞+「の名前/の正体/のこと」という所有格パターンも
# 自己言及として除外する
_SELF_REFERENCE_POSSESSIVE = re.compile(
    r"^(あなた|君|きみ|そちら|お前|あんた)(?:様|さま|さん)?の(?:名前|正体|こと)$"
)
# 2026-09-10(続き)発見: 「あなたにできないことって何?」が「あなた
# できますか?」という無関係なバラエティ番組にヒットし、偽の放送日程を
# ハルシネーションする不具合が見つかった。所有格(の名前等)だけでなく、
# 「あなたに/君が/お前は」のように**二人称代名詞で始まる質問文全体**は、
# 内容に関わらずLilas自身についての質問である可能性が高いので、
# 先頭が二人称代名詞かどうかで広く除外する
_SELF_REFERENCE_PREFIX = re.compile(
    r"^(あなた|君|きみ|そちら|お前|あんた|てめえ|貴殿|貴様)(?:様|さま|さん)?[はがにのでもって]"
)


def _is_self_reference(q: str) -> bool:
    if _SELF_REFERENCE_PREFIX.match(q):
        return True
    if _SELF_REFERENCE_POSSESSIVE.match(q):
        return True
    return _HONORIFIC_SUFFIX.sub("", q) in _SELF_REFERENCE_BASE


def _extract_search_query(text: str) -> tuple[str, str, re.Pattern | None] | None:
    for pattern, kind, detail_pattern in _KNOWLEDGE_TRIGGERS:
        m = pattern.search(text)
        if m:
            q = m.group(1).strip("、。 　")
            # 「じゃあ」「ところで」等の前置きを軽く除去
            q = re.sub(r"^(じゃあ|ところで|えっと|あの|なんか)、?", "", q)
            if not q:
                continue
            if _is_self_reference(q):
                continue
            # 2026-09-09: 「こんにちは、あなたは誰ですか?」のような挨拶+質問だと、
            # 正規表現が「こんにちは、あなた」まで丸ごと拾ってしまい、
            # Wikipediaで全く無関係な記事がヒットする不具合があった。
            # 挨拶語を含む・読点をまたぐ・長すぎる場合は、固有名詞の
            # 検索クエリとしては不自然なので除外する
            if any(g in q for g in _GREETING_WORDS):
                continue
            if "、" in q or len(q) > 20:
                continue
            # 2026-09-10: 「誰にも分からない」「どこにも無い」のような
            # 慣用句・反語表現は、疑問詞の直後に「にも」が続くことが多い。
            # Lilas自身への質問ではないので除外する
            if re.match(r"\s*にも", text[m.end() :]):
                continue
            return q, kind, detail_pattern
    return None


def _wikipedia_fact(query: str, detail_pattern: re.Pattern | None) -> tuple[str, str, str] | None:
    """Wikipediaで`query`を検索し、(記事タイトル, 冒頭の説明, 詳細の一文)を返す。

    2026-09-10: try_knowledgeとtry_comparisonの両方から使う共通処理として
    切り出した(以前はtry_knowledge内に直接書かれていた)。
    """
    try:
        search_resp = requests.get(
            "https://ja.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
            },
            headers=_WIKI_HEADERS,
            timeout=_TIMEOUT,
        )
        search_resp.raise_for_status()
        results = search_resp.json().get("query", {}).get("search", [])
        if not results:
            return None
        title = results[0]["title"]

        # 2026-09-09: 当初はREST APIの/page/summary/(記事冒頭の1段落のみ)を
        # 使っていたが、「東京タワーの高さ」等、具体的な数字が冒頭段落には
        # 無く「概要」節に初めて登場する記事が多いと判明した。
        # action=query&prop=extracts&explaintext で記事全文相当の
        # プレーンテキストを取得し、質問の種類に応じて該当する数字を含む
        # 一文を本文から追加で探す
        extract_resp = requests.get(
            "https://ja.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "titles": title,
                "format": "json",
            },
            headers=_WIKI_HEADERS,
            timeout=_TIMEOUT,
        )
        extract_resp.raise_for_status()
        pages = extract_resp.json().get("query", {}).get("pages", {})
        full_text = next(iter(pages.values()), {}).get("extract", "").strip()
        # 「== 地理 ==」のような節見出しが平文の中に混ざるとモデルが
        # 混乱しやすいので取り除く
        full_text = re.sub(r"={2,}\s*[^=\n]+\s*={2,}", "", full_text)
        # 2026-09-10発見: 一部のWikipedia記事(旧字体の人名等)に異体字
        # セレクタ(IVS、U+E0100-U+E01EF等)が含まれており、Windowsの
        # コンソール(cp932)に出力するとUnicodeEncodeErrorで丸ごと落ちる
        # 不具合があった。会話に不要な制御文字なので除去する
        full_text = re.sub(r"[\U0000fe00-\U0000fe0f\U000e0100-\U000e01ef]", "", full_text)
    except Exception:
        return None

    if not full_text:
        return None

    sentences = re.split(r"(?<=[。])", full_text)
    intro = "".join(sentences[:2]).strip()

    detail = ""
    if detail_pattern:
        # 本文の前半(参考文献・関連項目節などに入る前)だけを対象に、
        # 「数字+単位」のピンポイントな一致を探す。単なる単位語だけの
        # 一致だと「〜人には」のような無関係な文を拾ってしまうため
        # (2026-09-09発見)、必ず数字を伴うパターンにしている
        search_area = full_text[:6000]
        m = detail_pattern.search(search_area)
        if m and m.group(0) not in intro:
            # 一致箇所を含む一文だけを切り出す(前後の「。」区切りで探す)
            start = search_area.rfind("。", 0, m.start()) + 1
            end = search_area.find("。", m.end())
            end = end + 1 if end != -1 else m.end() + 30
            detail = search_area[start:end].strip()

    return title, intro, detail


def try_knowledge(text: str) -> str | None:
    """一般知識の質問を、Wikipedia日本語版APIで確実に答える。

    学習データに知識を丸暗記させる方式(training_log.md, 2026-09-09)は
    複製倍率を上げても簡単に上書きされることが分かったため、必要な時に
    都度調べる方式に切り替えた。APIキー不要のWikipedia日本語版のみ使用。
    """
    if not _looks_like_question(text):
        return None
    extracted = _extract_search_query(text)
    if not extracted:
        return None
    query, kind, detail_pattern = extracted

    fact = _wikipedia_fact(query, detail_pattern)
    if fact is None:
        return None
    title, intro, detail = fact
    short = (intro + " " + detail).strip()[:250]
    return f"Wikipediaで調べたところ、{title}について: {short}"


# 2026-09-10: 並列テストエージェントが、「AとB、どっちが高い?」のような
# 比較質問がそもそもツール層を一切通らず、モデルの生の記憶(丸暗記)だけで
# 答えていることを発見した。実際、訓練データに無い組み合わせでは
# 三択どころか二択の比較すら当てずっぽうになる(「富士山と富士山、どっちが
# 高い?」でも自信満々に答える)。電卓と同じ発想で、比較も実際の数値を
# 取得してPython側で確実に判定する
_COMPARISON_TRAITS: dict[str, tuple[str, re.Pattern]] = {
    "高い": ("height", re.compile(r"[\d,]+(?:\.\d+)?\s*(?:m|メートル)")),
    "低い": ("height", re.compile(r"[\d,]+(?:\.\d+)?\s*(?:m|メートル)")),
    "広い": ("area", re.compile(r"[\d,]+(?:\.\d+)?\s*(?:km2|km²|平方キロメートル)")),
    "大きい": ("area", re.compile(r"[\d,]+(?:\.\d+)?\s*(?:km2|km²|平方キロメートル)")),
    "人口が多い": ("population", re.compile(r"人口[^\d]{0,12}[\d,]{3,}\s*人")),
    "遠い": ("distance", re.compile(r"[\d,]+(?:\.\d+)?\s*km(?![²2])|[\d,]+(?:\.\d+)?\s*キロメートル")),
    "長い": ("distance", re.compile(r"[\d,]+(?:\.\d+)?\s*km(?![²2])|[\d,]+(?:\.\d+)?\s*キロメートル")),
}
_COMPARISON_PATTERN = re.compile(
    r"(.+?)と(.+?)(?:、|,)?\s*(?:どっちが|どちらが).{0,3}?(" + "|".join(_COMPARISON_TRAITS) + ")"
)
_INVERTED_TRAITS = {"低い", "小さい"}


def _first_number(text: str) -> float | None:
    m = re.search(r"[\d,]+(?:\.\d+)?", text)
    if not m:
        return None
    return float(m.group(0).replace(",", ""))


def _extract_value(text: str, detail_pattern: re.Pattern) -> float | None:
    """detail_patternに一致した箇所そのものから数字を取り出す。

    2026-09-10発見: 一文全体から「最初の数字」を拾うと、「1954年に
    測定され...8848 mという数値」のような文で、無関係な年号(1954)を
    高さの値として誤って掴んでしまう不具合があった。単位語(m/km2等)に
    隣接する数字だけをピンポイントで取り出す必要がある。
    """
    m = detail_pattern.search(text)
    if not m:
        return None
    return _first_number(m.group(0))


def try_comparison(text: str) -> str | None:
    """「AとB、どっちが高い?」を、両方の実際の数値を調べて確実に判定する。"""
    if not _looks_like_question(text):
        return None
    m = _COMPARISON_PATTERN.search(text)
    if not m:
        return None
    a_name, b_name, trait = m.group(1).strip("、。 "), m.group(2).strip("、。 "), m.group(3)
    if _is_self_reference(a_name) or _is_self_reference(b_name):
        return None
    if a_name == b_name:
        return None

    _, detail_pattern = _COMPARISON_TRAITS[trait]
    a_fact = _wikipedia_fact(a_name, detail_pattern)
    b_fact = _wikipedia_fact(b_name, detail_pattern)
    if a_fact is None or b_fact is None:
        return None

    a_title, a_intro, a_detail = a_fact
    b_title, b_intro, b_detail = b_fact
    a_val = _extract_value(a_detail, detail_pattern) or _extract_value(a_intro, detail_pattern)
    b_val = _extract_value(b_detail, detail_pattern) or _extract_value(b_intro, detail_pattern)
    if a_val is None or b_val is None:
        return None

    winner, winner_v, loser, loser_v = (
        (a_title, a_val, b_title, b_val) if a_val >= b_val else (b_title, b_val, a_title, a_val)
    )
    if trait in _INVERTED_TRAITS:
        winner, winner_v, loser, loser_v = loser, loser_v, winner, winner_v

    return (
        f"Wikipediaで調べたところ: {a_title}は{(a_detail or a_intro).strip()} "
        f"{b_title}は{(b_detail or b_intro).strip()} "
        f"比較すると{winner}の方が{trait}です({winner}: {winner_v:g}, {loser}: {loser_v:g})。"
    )


def try_tools(text: str) -> str | None:
    """calculator以外の全ツールを優先順位付きで試す。該当なければNone。

    try_comparisonはtry_knowledgeより先に試す(「AとBどっちが高い?」が
    「Aの高さ」のような単一事実パターンに部分一致して、知識ツールに
    横取りされるのを防ぐため)。
    """
    for fn in (try_datetime, try_holiday, try_weather, try_comparison, try_knowledge):
        result = fn(text)
        if result is not None:
            return result
    return None
