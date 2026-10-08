"""
複数ターン会話(callback形式)をテンプレート+スロット埋めで大量生成するスクリプト。

手書きの複数ターン会話(claude_authored_longcontext*.txt)は量が少なく
(合計30例程度)、文脈追跡が「特定の例の暗記」を超えて汎化しなかった。
そこで、話題の種類・具体的な事実(名前/日付/場所等)・雑談の挟み方・
呼び戻しの言い回しをそれぞれ変えながら大量に組み合わせ、「数ターン前の
事実を正しく参照する」というパターン自体をたくさんのバリエーションで
学習させることを狙う。

2026-09-09の根本原因調査: 最初のバージョン(v2_ft3)は各反応(intro_react/
detail_react/callback_q/callback_a)がトピックごとに固定の1文だったため、
persona_repeatで何十回も繰り返されるうちにその固定フレーズ自体を丸暗記し、
無関係な文脈にまで漏れ出す「テンプレート漏れ」が起きた。単純な語句の
言い換え(v2_ft4)だけでは文構造自体の単調さが解消できず、効果は不十分
だった。今回は反応そのものを複数パターン用意し、生成のたびにランダムに
選ぶことで、根本的に単調さを減らす。

使い方:
    python scripts/generate_longcontext_data.py
    -> index/generated_longcontext_ja.txt を書き出す
"""

from __future__ import annotations

import random
from pathlib import Path

random.seed(7)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "index" / "generated_longcontext_ja.txt"

blocks: list[str] = []

# ---- 汎用の反応プール(トピック固有の固有名詞を含まない、複数パターン) ----
# 「楽しみ・わくわく系」「心配・気遣い系」「感心・応援系」の3系統。
# 各トピックのintro_react/detail_reactはこのプールからランダムに選ぶ。

REACT_EXCITED = [
    "それは楽しみですね!",
    "それは楽しみです、わくわくしますね。",
    "素敵なお話ですね、楽しみが増えますね。",
    "それは待ち遠しいですね。",
    "聞いているだけでこちらまで嬉しくなりますね。",
]

REACT_CONCERNED = [
    "それはご心配ですね。",
    "それは大変そうですね、無理なさらないでくださいね。",
    "それは心配になりますね、お大事になさってください。",
    "そうでしたか、お気をつけくださいね。",
]

REACT_ADMIRING = [
    "それは素晴らしいですね、応援しております。",
    "素敵な心がけですね。",
    "それは立派ですね、頑張っていらっしゃるのですね。",
    "とても良い取り組みですね。",
]

REACT_CURIOUS = [
    "それは興味深いですね、詳しく伺いたいです。",
    "気になりますね、もう少し聞かせてください。",
    "それは楽しそうな話ですね。",
]


def _react(pool: list[str]) -> str:
    return random.choice(pool)


# ---- 話題ごとの「事実の導入」「呼び戻し質問(複数パターン)」----
# intro/detailは事実そのものなのでトピック固有の1文のまま、
# reactは上の汎用プールから、callback_q/callback_aは各3パターン用意する。

topics = [
    {
        "intro": "今度{dest}に旅行に行くことになったんだ",
        "intro_pool": REACT_EXCITED,
        "detail": "{date}に行く予定なんだ",
        "detail_pool": REACT_ADMIRING,
        "callback_q": [
            "{callback}旅行の件だけど、宿はもう決めた?",
            "{callback}{dest}への旅行、準備は進んでる?",
            "{callback}例の旅行、あれからどうなった?",
        ],
        "callback_a": [
            "{date}に{dest}へ行かれるご旅行ですね、宿泊先が決まるとさらに楽しみになりますね。",
            "{dest}へのご旅行のお話でしたね、{date}のご予定でしたか、準備はいかがですか?",
            "楽しみにしていらっしゃった{dest}旅行ですね、その後の進み具合はいかがでしょうか?",
        ],
        "slots": [("dest", "沖縄"), ("date", "来月")],
    },
    {
        "intro": "うちの{pet}の名前、{name}っていうんだ",
        "intro_pool": REACT_EXCITED,
        "detail": "{age}なんだ",
        "detail_pool": REACT_CURIOUS,
        "callback_q": [
            "{callback}{pet}の{name}だけど、最近元気にしてる?",
            "{callback}{name}ちゃん、その後どうしてる?",
            "{callback}{pet}の様子、変わりない?",
        ],
        "callback_a": [
            "{age}の{name}ちゃんですね、お元気だとよいのですが、いかがですか?",
            "{name}ちゃんのお話でしたね、{pet}ちゃんは今も元気にしていますか?",
            "可愛らしい{name}ちゃんですね、その後の様子はいかがでしょうか?",
        ],
        "slots": [("pet", "猫"), ("name", "モモ"), ("age", "2歳")],
    },
    {
        "intro": "今度{event}があるんだ",
        "intro_pool": REACT_ADMIRING,
        "detail": "{date}に控えてるんだ",
        "detail_pool": REACT_CONCERNED,
        "callback_q": [
            "{callback}{event}の準備、進んでる?",
            "{callback}例の{event}、その後どう?",
            "{callback}{date}の{event}だけど、心の準備はできてる?",
        ],
        "callback_a": [
            "{date}に控えていらっしゃる{event}ですね、順調に進んでいらっしゃいますか?",
            "{event}のお話でしたね、準備の方はいかがですか?",
            "大事な{event}でしたね、落ち着いて臨めそうですか?",
        ],
        "slots": [("event", "面接"), ("date", "来週")],
    },
    {
        "intro": "最近{hobby}を始めたんだ",
        "intro_pool": REACT_ADMIRING,
        "detail": "{freq}くらいのペースでやってるよ",
        "detail_pool": REACT_ADMIRING,
        "callback_q": [
            "{callback}{hobby}、続けられてる?",
            "{callback}例の{hobby}、その後どう?",
            "{callback}{hobby}は今も{freq}くらいのペースなの?",
        ],
        "callback_a": [
            "{freq}のペースで続けていらっしゃる{hobby}ですね、その調子で頑張ってくださいね。",
            "{hobby}のお話でしたね、無理なく続けられていますか?",
            "始められた{hobby}ですね、楽しく続けられていますか?",
        ],
        "slots": [("hobby", "ヨガ"), ("freq", "週に2回")],
    },
    {
        "intro": "{item}を買おうか迷ってるんだ",
        "intro_pool": REACT_CURIOUS,
        "detail": "{color}にしようか考えてる",
        "detail_pool": REACT_EXCITED,
        "callback_q": [
            "{callback}{item}、結局買った?",
            "{callback}例の{item}の件、決まった?",
            "{callback}{color}の{item}だけど、どうすることにした?",
        ],
        "callback_a": [
            "{color}の{item}のお話でしたね、購入されましたか?",
            "{item}を検討されていましたね、決心はつきましたか?",
            "迷っていらっしゃった{item}ですね、その後いかがでしたか?",
        ],
        "slots": [("item", "自転車"), ("color", "青")],
    },
    {
        "intro": "{subject}の勉強を頑張ってるんだ",
        "intro_pool": REACT_ADMIRING,
        "detail": "{goal}を目指してるんだ",
        "detail_pool": REACT_ADMIRING,
        "callback_q": [
            "{callback}{subject}の勉強、順調?",
            "{callback}例の{subject}、その後どう?",
            "{callback}{goal}に向けて、進み具合はどう?",
        ],
        "callback_a": [
            "{goal}を目指していらっしゃる{subject}の勉強ですね、順調に進んでいらっしゃいますか?",
            "{subject}のお話でしたね、コツコツ続けられていますか?",
            "{goal}という目標でしたね、手応えはいかがですか?",
        ],
        "slots": [("subject", "英語"), ("goal", "TOEIC800点")],
    },
    {
        "intro": "{family}が今度{life_event}するんだ",
        "intro_pool": REACT_EXCITED,
        "detail": "{date}に予定してるみたい",
        "detail_pool": REACT_EXCITED,
        "callback_q": [
            "{callback}{family}の{life_event}の件、詳細決まった?",
            "{callback}例の{life_event}、その後どうなった?",
            "{callback}{date}の{life_event}だけど、準備は進んでる?",
        ],
        "callback_a": [
            "{date}に予定されている{family}様の{life_event}ですね、詳細が決まるといいですね。",
            "{life_event}のお話でしたね、準備は進んでいますか?",
            "おめでたい{life_event}でしたね、その後いかがですか?",
        ],
        "slots": [("family", "妹"), ("life_event", "結婚"), ("date", "来年の春")],
    },
    {
        "intro": "最近{symptom}が続いてるんだ",
        "intro_pool": REACT_CONCERNED,
        "detail": "{action}してみようと思ってる",
        "detail_pool": REACT_CONCERNED,
        "callback_q": [
            "{callback}体調の件だけど、その後どう?",
            "{callback}{symptom}、よくなった?",
            "{callback}体の調子、変わりない?",
        ],
        "callback_a": [
            "{symptom}のこと、心配しておりました。{action}された後、いかがですか?",
            "体調のお話でしたね、{symptom}はよくなりましたか?",
            "ご心配していた{symptom}ですね、無理せずお過ごしですか?",
        ],
        "slots": [("symptom", "頭痛"), ("action", "早めに休む")],
    },
    {
        "intro": "水槽で{fish}を飼い始めたんだ",
        "intro_pool": REACT_EXCITED,
        "detail": "{count}匹いて、名前は{name}っていうんだ",
        "detail_pool": REACT_CURIOUS,
        "callback_q": [
            "{callback}{fish}の{name}だけど、水槽の掃除はもうした?",
            "{callback}{name}、その後元気にしてる?",
            "{callback}水槽の件、あれからどうなった?",
        ],
        "callback_a": [
            "{name}ちゃんの水槽のお話でしたね、掃除は済まされましたか?",
            "{fish}の{name}ちゃんですね、その後変わりないですか?",
            "水槽のお話でしたね、{name}ちゃんは元気にしていますか?",
        ],
        "slots": [("fish", "金魚"), ("count", "3"), ("name", "スイ")],
    },
    {
        "intro": "今{book}っていう本を読んでるんだ",
        "intro_pool": REACT_CURIOUS,
        "detail": "{genre}のジャンルで、今{progress}くらい読み進めたよ",
        "detail_pool": REACT_ADMIRING,
        "callback_q": [
            "{callback}あの本、結局読み終わった?",
            "{callback}{book}、その後どこまで読んだ?",
            "{callback}例の本の話だけど、面白かった?",
        ],
        "callback_a": [
            "{book}のお話でしたね、{genre}のジャンルの。読み終わりましたか?",
            "読んでいらっしゃった{book}ですね、その後の進み具合はいかがですか?",
            "{genre}の{book}でしたね、楽しんで読めていますか?",
        ],
        "slots": [("book", "星の旅人"), ("genre", "ミステリー"), ("progress", "半分")],
    },
    {
        "intro": "職場で{project}っていうプロジェクトを任されたんだ",
        "intro_pool": REACT_ADMIRING,
        "detail": "{deadline}が締め切りで、少し忙しくなりそう",
        "detail_pool": REACT_CONCERNED,
        "callback_q": [
            "{callback}{project}の件、進み具合はどう?",
            "{callback}例のプロジェクト、その後どう?",
            "{callback}{deadline}の締め切り、間に合いそう?",
        ],
        "callback_a": [
            "{deadline}が締め切りの{project}ですね、順調に進んでいらっしゃいますか?",
            "{project}のお話でしたね、忙しさは落ち着きましたか?",
            "大事な{project}でしたね、無理なさっていませんか?",
        ],
        "slots": [("project", "新商品の企画"), ("deadline", "今月末")],
    },
    {
        "intro": "週末に{dish}を作ってみようと思ってるんだ",
        "intro_pool": REACT_EXCITED,
        "detail": "{source}で見つけたレシピを参考にするつもりなんだ",
        "detail_pool": REACT_CURIOUS,
        "callback_q": [
            "{callback}{dish}、作ってみてどうだった?",
            "{callback}例の料理、うまくいった?",
            "{callback}{dish}の件、結局作った?",
        ],
        "callback_a": [
            "{source}のレシピで作られる予定だった{dish}ですね、うまくいきましたか?",
            "{dish}のお話でしたね、美味しくできましたか?",
            "楽しみにしていらっしゃった{dish}ですね、その後いかがでしたか?",
        ],
        "slots": [("dish", "カレー"), ("source", "料理サイト")],
    },
    {
        "intro": "{vehicle}を買い替えようか検討してるんだ",
        "intro_pool": REACT_CURIOUS,
        "detail": "{brand}のものが気になってるんだ",
        "detail_pool": REACT_EXCITED,
        "callback_q": [
            "{callback}{vehicle}の件、結局どうすることにした?",
            "{callback}例の{vehicle}、決まった?",
            "{callback}{brand}の{vehicle}、買うことにした?",
        ],
        "callback_a": [
            "{brand}のものを検討されていた{vehicle}のお話ですね、決まりましたか?",
            "{vehicle}の買い替えのお話でしたね、その後いかがですか?",
            "迷っていらっしゃった{vehicle}ですね、決心はつきましたか?",
        ],
        "slots": [("vehicle", "自動車"), ("brand", "国産メーカー")],
    },
    {
        "intro": "{plant}を育て始めたんだ",
        "intro_pool": REACT_EXCITED,
        "detail": "{care}くらいの頻度で水やりをしてるよ",
        "detail_pool": REACT_ADMIRING,
        "callback_q": [
            "{callback}{plant}、元気に育ってる?",
            "{callback}例の{plant}、その後どう?",
            "{callback}{plant}の様子、変わりない?",
        ],
        "callback_a": [
            "{care}の頻度でお世話されている{plant}ですね、順調に育っていますか?",
            "{plant}のお話でしたね、元気に育っていますか?",
            "育てていらっしゃる{plant}ですね、その後変わりありませんか?",
        ],
        "slots": [("plant", "観葉植物"), ("care", "週に2回")],
    },
    {
        "intro": "最近{diet}を意識してるんだ",
        "intro_pool": REACT_ADMIRING,
        "detail": "{goal}を目標にしてるんだ",
        "detail_pool": REACT_ADMIRING,
        "callback_q": [
            "{callback}{diet}の件、続けられてる?",
            "{callback}例の{diet}、その後どう?",
            "{callback}{goal}、近づいてきてる?",
        ],
        "callback_a": [
            "{goal}を目標にされていた{diet}ですね、順調に続けられていますか?",
            "{diet}のお話でしたね、無理なく続けられていますか?",
            "{goal}という目標でしたね、手応えはいかがですか?",
        ],
        "slots": [("diet", "食事管理"), ("goal", "3kg減量")],
    },
    {
        "intro": "来月{room}のある部屋に引っ越すことになったんだ",
        "intro_pool": REACT_EXCITED,
        "detail": "{furniture}を新しく揃えようと思ってるんだ",
        "detail_pool": REACT_EXCITED,
        "callback_q": [
            "{callback}引っ越しの件、{furniture}はもう決めた?",
            "{callback}例の引っ越し、準備は進んでる?",
            "{callback}{room}のお部屋、その後どう?",
        ],
        "callback_a": [
            "{room}のあるお部屋への引っ越しでしたね、{furniture}は決まりましたか?",
            "引っ越しのお話でしたね、準備は進んでいますか?",
            "新生活のご準備でしたね、{furniture}は決まりましたか?",
        ],
        "slots": [("room", "書斎"), ("furniture", "本棚")],
    },
    {
        "intro": "うちの{pet2}を{vetreason}で病院に連れて行ったんだ",
        "intro_pool": REACT_CONCERNED,
        "detail": "{result}って言われて、少し安心したよ",
        "detail_pool": REACT_ADMIRING,
        "callback_q": [
            "{callback}{pet2}の体調だけど、その後どう?",
            "{callback}{pet2}、元気にしてる?",
            "{callback}病院の件、あれからどう?",
        ],
        "callback_a": [
            "{vetreason}で病院に行かれた{pet2}ちゃんですね、{result}とのことで安心しました。今の様子はいかがですか?",
            "{pet2}ちゃんのお話でしたね、その後変わりないですか?",
            "ご心配だった{pet2}ちゃんですね、元気にしていますか?",
        ],
        "slots": [("pet2", "うさぎ"), ("vetreason", "定期検診"), ("result", "特に異常なし")],
    },
    {
        "intro": "{subscription}に加入しようか迷ってるんだ",
        "intro_pool": REACT_CURIOUS,
        "detail": "{price}くらいなら続けられそうだと思ってる",
        "detail_pool": REACT_ADMIRING,
        "callback_q": [
            "{callback}{subscription}の件、結局加入した?",
            "{callback}例のサービス、どうすることにした?",
            "{callback}{subscription}、使ってみてどう?",
        ],
        "callback_a": [
            "{price}のプランを検討されていた{subscription}のお話でしたね、加入されましたか?",
            "{subscription}のお話でしたね、決心はつきましたか?",
            "迷っていらっしゃった{subscription}ですね、その後いかがですか?",
        ],
        "slots": [("subscription", "動画配信サービス"), ("price", "月1000円")],
    },
    # 2026-09-09追加: v2_ft5で「訓練済みトピックに似た形にしか汎化しない」
    # ことが判明したため、対人・職場系など構造の異なるトピックを増やす。
    {
        "intro": "来週{party}があるんだ",
        "intro_pool": REACT_EXCITED,
        "detail": "{purpose}の会でね",
        "detail_pool": REACT_ADMIRING,
        "callback_q": [
            "{callback}{party}の件だけど、場所はもう決まった?",
            "{callback}例の{party}、準備はどう?",
            "{callback}{party}、結局どうなった?",
        ],
        "callback_a": [
            "{purpose}の{party}でしたね、場所は決まりましたか?",
            "{party}のお話でしたね、準備は進んでいますか?",
            "楽しみにしていらっしゃった{party}ですね、その後いかがですか?",
        ],
        "slots": [("party", "部署の歓迎会"), ("purpose", "新しく入った後輩を歓迎する")],
    },
    {
        "intro": "{colleague}と{meeting}をすることになったんだ",
        "intro_pool": REACT_ADMIRING,
        "detail": "{date}の予定なんだ",
        "detail_pool": REACT_CURIOUS,
        "callback_q": [
            "{callback}{meeting}の件、無事終わった?",
            "{callback}例の{meeting}、どうだった?",
            "{callback}{colleague}との{meeting}、準備はできてる?",
        ],
        "callback_a": [
            "{date}にご予定されていた{meeting}ですね、いかがでしたか?",
            "{colleague}様との{meeting}のお話でしたね、無事終わりましたか?",
            "{meeting}のお話でしたね、準備は整いましたか?",
        ],
        "slots": [("colleague", "取引先の方"), ("meeting", "打ち合わせ"), ("date", "来週火曜")],
    },
    {
        "intro": "今度{friend}の誕生日プレゼントを選ぼうと思ってるんだ",
        "intro_pool": REACT_EXCITED,
        "detail": "{gift}にしようか迷ってる",
        "detail_pool": REACT_CURIOUS,
        "callback_q": [
            "{callback}{friend}へのプレゼント、結局何にした?",
            "{callback}例のプレゼントの件、決まった?",
            "{callback}{gift}にするか迷ってたやつ、どうなった?",
        ],
        "callback_a": [
            "{gift}にするか迷っていらっしゃった{friend}様へのプレゼントですね、決まりましたか?",
            "プレゼントのお話でしたね、{friend}様に喜んでいただけそうですか?",
            "迷っていらっしゃったプレゼントですね、決心はつきましたか?",
        ],
        "slots": [("friend", "友人"), ("gift", "マグカップ")],
    },
    {
        "intro": "{task}を{colleague2}に引き継ぐことになったんだ",
        "intro_pool": REACT_CURIOUS,
        "detail": "{date2}までに資料をまとめないといけないんだ",
        "detail_pool": REACT_CONCERNED,
        "callback_q": [
            "{callback}{task}の引き継ぎ、終わった?",
            "{callback}例の引き継ぎの件、うまくいってる?",
            "{callback}{colleague2}への引き継ぎ、進んでる?",
        ],
        "callback_a": [
            "{date2}までのご予定だった{task}の引き継ぎですね、うまくいきましたか?",
            "{colleague2}様への引き継ぎのお話でしたね、資料はまとまりましたか?",
            "引き継ぎのお話でしたね、無事に終わりましたか?",
        ],
        "slots": [("task", "担当業務"), ("colleague2", "後任の方"), ("date2", "今週末")],
    },
    {
        "intro": "{friend2}と今度{meal}に行く約束をしたんだ",
        "intro_pool": REACT_EXCITED,
        "detail": "{place}にしようか話してるところなんだ",
        "detail_pool": REACT_EXCITED,
        "callback_q": [
            "{callback}{friend2}との{meal}、お店決まった?",
            "{callback}例の{meal}の約束、日程決まった?",
            "{callback}{meal}の件、楽しみにしてる?",
        ],
        "callback_a": [
            "{place}にするか話していらっしゃった{meal}ですね、決まりましたか?",
            "{friend2}様との{meal}のお話でしたね、楽しみですね。",
            "{meal}のご予定でしたね、詳細は決まりましたか?",
        ],
        "slots": [("friend2", "学生時代の友人"), ("meal", "食事"), ("place", "駅前のお店")],
    },
    {
        "intro": "使っていない{service}を解約しようか迷ってるんだ",
        "intro_pool": REACT_CURIOUS,
        "detail": "{reason}からなんだ",
        "detail_pool": REACT_ADMIRING,
        "callback_q": [
            "{callback}{service}の件、結局解約した?",
            "{callback}例のサービスの解約、どうなった?",
            "{callback}{service}、まだ迷ってる?",
        ],
        "callback_a": [
            "{reason}で迷っていらっしゃった{service}ですね、解約されましたか?",
            "{service}のお話でしたね、決心はつきましたか?",
            "迷っていらっしゃった{service}ですね、その後どうされましたか?",
        ],
        "slots": [("service", "ジムの会員"), ("reason", "最近通えていない")],
    },
    # 2026-09-10追加: 並列テストエージェントが「訓練済みトピックと
    # 構造が異なる話題では、名前ではなく別のテンプレートに間違って
    # 当てはめる」現象を発見した(トマトの害虫相談にペットの健康診断
    # テンプレートで返す等)。話題の「形」自体をさらに広げる。
    # また「数字・時刻の呼び戻しは新旧問わず全滅」という発見を踏まえ、
    # callback_aで具体的な数字・時刻を明示的に再言及するトピックを増やす。
    {
        "intro": "{place}に傘を忘れてきちゃったんだ",
        "intro_pool": REACT_CONCERNED,
        "detail": "{color}の傘だから、見つかるといいんだけど",
        "detail_pool": REACT_CONCERNED,
        "callback_q": [
            "{callback}傘の件、結局見つかった?",
            "{callback}{place}に忘れた傘、どうなった?",
            "{callback}{color}の傘、見つかった?",
        ],
        "callback_a": [
            "{place}に忘れられた{color}の傘のお話でしたね、見つかりましたか?",
            "傘のお話でしたね、{place}に問い合わせてみましたか?",
            "{color}の傘でしたね、その後見つかりましたか?",
        ],
        "slots": [("place", "カフェ"), ("color", "青")],
    },
    {
        "intro": "育ててる{plant2}に{pest}がついちゃったんだ",
        "intro_pool": REACT_CONCERNED,
        "detail": "{action2}で対処しようと思ってる",
        "detail_pool": REACT_ADMIRING,
        "callback_q": [
            "{callback}{plant2}の{pest}、その後どう?",
            "{callback}例の害虫の件、良くなった?",
            "{callback}{plant2}、元気になった?",
        ],
        "callback_a": [
            "{pest}がついていた{plant2}のお話でしたね、{action2}で良くなりましたか?",
            "{plant2}のお話でしたね、その後の様子はいかがですか?",
            "ご心配だった{plant2}ですね、元気になってきましたか?",
        ],
        "slots": [("plant2", "トマト"), ("pest", "アブラムシ"), ("action2", "薬をまく")],
    },
    {
        "intro": "うちの{pet3}が大事にしてる{toy}が見当たらないんだ",
        "intro_pool": REACT_CONCERNED,
        "detail": "{place2}を探してみたんだけど無くて",
        "detail_pool": REACT_CONCERNED,
        "callback_q": [
            "{callback}{pet3}の{toy}、見つかった?",
            "{callback}例のおもちゃの件、どうなった?",
            "{callback}{toy}、結局見つかった?",
        ],
        "callback_a": [
            "{pet3}ちゃんの{toy}のお話でしたね、見つかりましたか?",
            "{toy}を探していらっしゃいましたね、{place2}以外の場所も見てみましたか?",
            "見当たらなかった{toy}ですね、その後いかがですか?",
        ],
        "slots": [("pet3", "犬"), ("toy", "お気に入りのボール"), ("place2", "庭")],
    },
    {
        "intro": "{device}を修理に出したんだ",
        "intro_pool": REACT_CONCERNED,
        "detail": "{repair_time}で仕上がるって言われたよ",
        "detail_pool": REACT_ADMIRING,
        "callback_q": [
            "{callback}{device}の修理、結局いつ仕上がるんだっけ?",
            "{callback}修理の件、いつ受け取れる予定?",
            "{callback}{device}、もう戻ってきた?",
        ],
        "callback_a": [
            "{repair_time}で仕上がるとのことでしたね、その予定通りでしょうか?",
            "{device}の修理のお話でしたね、{repair_time}という話でしたね。",
            "修理のお話でしたね、{repair_time}のご予定でしたね。",
        ],
        "slots": [("device", "スマホ"), ("repair_time", "3日後")],
    },
    {
        "intro": "{company_a}と{company_b}、2つの仕事のオファーで迷ってるんだ",
        "intro_pool": REACT_CURIOUS,
        "detail": "{deadline2}までに決めないといけないんだ",
        "detail_pool": REACT_CONCERNED,
        "callback_q": [
            "{callback}仕事の件、結局どっちにするか決めた?",
            "{callback}{company_a}と{company_b}の件、どうなった?",
            "{callback}オファーの件、期限までに決められそう?",
        ],
        "callback_a": [
            "{company_a}様と{company_b}様で迷っていらっしゃったお話でしたね、決められましたか?",
            "{deadline2}までのご予定でしたね、決心はつきましたか?",
            "大事な決断でしたね、その後いかがですか?",
        ],
        "slots": [("company_a", "A社"), ("company_b", "B社"), ("deadline2", "今週末")],
    },
    {
        "intro": "{item2}の荷物が届くのを待ってるんだ",
        "intro_pool": REACT_EXCITED,
        "detail": "{arrival_date}に届く予定になってるよ",
        "detail_pool": REACT_EXCITED,
        "callback_q": [
            "{callback}{item2}の荷物、結局いつ届くんだっけ?",
            "{callback}例の荷物、もう届いた?",
            "{callback}{item2}、届く予定は変わってない?",
        ],
        "callback_a": [
            "{arrival_date}に届くご予定でしたね、その通り届きましたか?",
            "{item2}の荷物のお話でしたね、{arrival_date}のご予定でしたね。",
            "楽しみにされていた{item2}ですね、{arrival_date}に届く予定でしたね。",
        ],
        "slots": [("item2", "新しいカバン"), ("arrival_date", "明後日")],
    },
    {
        "intro": "{person}のためにサプライズを計画してるんだ",
        "intro_pool": REACT_EXCITED,
        "detail": "{event_date}に実行する予定なんだ",
        "detail_pool": REACT_EXCITED,
        "callback_q": [
            "{callback}サプライズの件、結局いつやるんだっけ?",
            "{callback}{person}へのサプライズ、準備は進んでる?",
            "{callback}例の計画、予定通り進んでる?",
        ],
        "callback_a": [
            "{event_date}に実行されるご予定でしたね、準備は進んでいますか?",
            "{person}様へのサプライズでしたね、{event_date}のご予定でしたね。",
            "楽しみにされていたサプライズですね、{event_date}に控えていますね。",
        ],
        "slots": [("person", "友人"), ("event_date", "来週土曜")],
    },
    {
        "intro": "{furniture2}の値段を{price_from}円から{price_to}円に交渉できたんだ",
        "intro_pool": REACT_EXCITED,
        "detail": "思っていたよりうまくいったよ",
        "detail_pool": REACT_ADMIRING,
        "callback_q": [
            "{callback}{furniture2}の件、結局いくらになったんだっけ?",
            "{callback}あの交渉、最終的にいくらだった?",
            "{callback}{furniture2}、もう手元に届いた?",
        ],
        "callback_a": [
            "{price_from}円から{price_to}円に交渉されたのでしたね、うまくいって良かったです。",
            "{furniture2}のお話でしたね、{price_to}円になったのでしたね。",
            "交渉されていた{furniture2}ですね、{price_to}円で決まったのでしたね。",
        ],
        "slots": [("furniture2", "ソファ"), ("price_from", "50000"), ("price_to", "40000")],
    },
]

distractors = [
    ("今日の天気すごくいいね", "それはいいですね、気持ちのよい一日になりそうですね。"),
    ("さっきコーヒーをこぼしちゃって大変だった", "それは大変でしたね、火傷などございませんでしたか?"),
    ("最近見た映画がすごく面白かったんだ", "それは楽しそうですね、どんな映画だったのですか?"),
    ("今日のお昼、美味しいラーメン食べたよ", "それはよろしいですね、お腹も満たされたのではないでしょうか。"),
    ("週末は友達と買い物に行く予定なんだ", "それは楽しみですね。"),
    ("最近ちょっと忙しくてバタバタしてるんだ", "お疲れさまです、無理なさらないでくださいね。"),
    ("好きな音楽の話をしてもいい?", "はい、ぜひ聞かせてください。"),
    ("今日は早起きできたんだ", "それは素晴らしいですね、気持ちの良い一日の始まりですね。"),
    ("さっき近所の犬に吠えられてびっくりした", "それは驚きましたね、お怪我はありませんでしたか?"),
    ("今日は電車が少し遅れて焦った", "それは大変でしたね、間に合いましたか?"),
    ("お気に入りのカフェが新しくオープンしたんだ", "それは楽しみですね、今度行ってみたいですね。"),
    ("最近寝つきが悪くて困ってるんだ", "それはご心配ですね、無理なさらないでくださいね。"),
    ("友達から美味しいお菓子をもらったんだ", "それは嬉しいですね、どんなお菓子だったのですか?"),
    ("今日は部屋の掃除を頑張ったよ", "それは気持ちよさそうですね、お疲れさまでした。"),
]

callback_phrases = [
    "そういえば、", "ところで、", "この前話してた", "さっき言ってた",
    "前に話した", "そうそう、", "話は変わるけど、",
]


def build_dialogue(topic: dict) -> str:
    slots = dict(topic["slots"])
    lines = []

    intro = topic["intro"].format(**slots)
    lines.append(f"User: {intro}")
    lines.append(f"Lilas: {_react(topic['intro_pool'])}")

    detail = topic["detail"].format(**slots)
    lines.append(f"User: {detail}")
    lines.append(f"Lilas: {_react(topic['detail_pool'])}")

    # 雑談を1〜2個挟んで、直前の話題から距離を稼ぐ
    for du, da in random.sample(distractors, k=random.choice([1, 2])):
        lines.append(f"User: {du}")
        lines.append(f"Lilas: {da}")

    callback = random.choice(callback_phrases)
    cb_slots = dict(slots)
    cb_slots["callback"] = callback
    q = random.choice(topic["callback_q"]).format(**cb_slots)
    a = random.choice(topic["callback_a"]).format(**cb_slots)
    lines.append(f"User: {q}")
    lines.append(f"Lilas: {a}")

    return "\n".join(lines)


# 各トピックについて、スロットの値を変えながら複数バリエーション作る
slot_variants = {
    0: [[("dest", "沖縄"), ("date", "来月")], [("dest", "北海道"), ("date", "再来週")],
        [("dest", "京都"), ("date", "来週末")], [("dest", "福岡"), ("date", "来月半ば")]],
    1: [[("pet", "猫"), ("name", "モモ"), ("age", "2歳")],
        [("pet", "犬"), ("name", "ハチ"), ("age", "4歳")],
        [("pet", "うさぎ"), ("name", "ユキ"), ("age", "1歳")],
        [("pet", "犬"), ("name", "コロ"), ("age", "5歳")]],
    2: [[("event", "面接"), ("date", "来週")], [("event", "プレゼン"), ("date", "明後日")],
        [("event", "資格試験"), ("date", "来月")], [("event", "発表会"), ("date", "今週末")]],
    3: [[("hobby", "ヨガ"), ("freq", "週に2回")], [("hobby", "ジョギング"), ("freq", "毎朝")],
        [("hobby", "編み物"), ("freq", "週末だけ")], [("hobby", "水泳"), ("freq", "週に1回")]],
    4: [[("item", "自転車"), ("color", "青")], [("item", "腕時計"), ("color", "黒")],
        [("item", "カバン"), ("color", "茶色")], [("item", "パソコン"), ("color", "白")]],
    5: [[("subject", "英語"), ("goal", "TOEIC800点")], [("subject", "簿記"), ("goal", "2級合格")],
        [("subject", "プログラミング"), ("goal", "資格取得")], [("subject", "フランス語"), ("goal", "日常会話")]],
    6: [[("family", "妹"), ("life_event", "結婚"), ("date", "来年の春")],
        [("family", "兄"), ("life_event", "引っ越し"), ("date", "来月")],
        [("family", "娘"), ("life_event", "入学"), ("date", "来年の4月")],
        [("family", "息子"), ("life_event", "卒業"), ("date", "来年の3月")]],
    7: [[("symptom", "頭痛"), ("action", "早めに休む")], [("symptom", "咳"), ("action", "病院に行く")],
        [("symptom", "肩こり"), ("action", "マッサージに行く")], [("symptom", "腰痛"), ("action", "安静にする")]],
    8: [[("fish", "金魚"), ("count", "3"), ("name", "スイ")],
        [("fish", "メダカ"), ("count", "10"), ("name", "ヒカル")],
        [("fish", "熱帯魚"), ("count", "5"), ("name", "ネモ")],
        [("fish", "金魚"), ("count", "2"), ("name", "アカ")],
        [("fish", "メダカ"), ("count", "6"), ("name", "ソラ")]],
    9: [[("book", "星の旅人"), ("genre", "ミステリー"), ("progress", "半分")],
        [("book", "海辺の記憶"), ("genre", "恋愛小説"), ("progress", "3分の1")],
        [("book", "灯台の物語"), ("genre", "ファンタジー"), ("progress", "7割")],
        [("book", "夜明けの街"), ("genre", "SF"), ("progress", "3分の2")],
        [("book", "風の記録"), ("genre", "歴史小説"), ("progress", "4分の1")]],
    10: [[("project", "新商品の企画"), ("deadline", "今月末")],
         [("project", "社内システムの改修"), ("deadline", "来週")],
         [("project", "販促キャンペーン"), ("deadline", "来月頭")],
         [("project", "新人研修の資料作成"), ("deadline", "今週末")],
         [("project", "顧客アンケートの集計"), ("deadline", "明日")]],
    11: [[("dish", "カレー"), ("source", "料理サイト")],
         [("dish", "グラタン"), ("source", "母のレシピ")],
         [("dish", "パスタ"), ("source", "料理番組")],
         [("dish", "肉じゃが"), ("source", "料理本")],
         [("dish", "手作りパン"), ("source", "動画サイト")]],
    12: [[("vehicle", "自動車"), ("brand", "国産メーカー")],
         [("vehicle", "自転車"), ("brand", "スポーツタイプ")],
         [("vehicle", "バイク"), ("brand", "外国メーカー")],
         [("vehicle", "電動アシスト自転車"), ("brand", "軽量モデル")]],
    13: [[("plant", "観葉植物"), ("care", "週に2回")],
         [("plant", "多肉植物"), ("care", "月に2回")],
         [("plant", "バラ"), ("care", "毎日")],
         [("plant", "ハーブ"), ("care", "2日に1回")]],
    14: [[("diet", "食事管理"), ("goal", "3kg減量")],
         [("diet", "筋トレ"), ("goal", "体力アップ")],
         [("diet", "ウォーキング"), ("goal", "1日8000歩")],
         [("diet", "糖質制限"), ("goal", "健康診断の数値改善")]],
    15: [[("room", "書斎"), ("furniture", "本棚")],
         [("room", "広いキッチン"), ("furniture", "食器棚")],
         [("room", "ベランダ"), ("furniture", "物干し")],
         [("room", "収納の多い部屋"), ("furniture", "クローゼット")]],
    16: [[("pet2", "うさぎ"), ("vetreason", "定期検診"), ("result", "特に異常なし")],
         [("pet2", "猫"), ("vetreason", "ワクチン接種"), ("result", "問題なし")],
         [("pet2", "犬"), ("vetreason", "足を痛がっていたので"), ("result", "軽い捻挫だけ")],
         [("pet2", "フェレット"), ("vetreason", "食欲不振"), ("result", "少し様子見でよいとのこと")]],
    17: [[("subscription", "動画配信サービス"), ("price", "月1000円")],
         [("subscription", "音楽配信サービス"), ("price", "月800円")],
         [("subscription", "電子書籍の読み放題"), ("price", "月1500円")],
         [("subscription", "オンラインヨガ教室"), ("price", "月3000円")]],
    18: [[("party", "部署の歓迎会"), ("purpose", "新しく入った後輩を歓迎する")],
         [("party", "サークルの送別会"), ("purpose", "卒業する先輩を送る")],
         [("party", "町内会の夏祭り"), ("purpose", "地域の人たちが集まる")],
         [("party", "友人の結婚祝いの集まり"), ("purpose", "お祝いをする")]],
    19: [[("colleague", "取引先の方"), ("meeting", "打ち合わせ"), ("date", "来週火曜")],
         [("colleague", "上司"), ("meeting", "面談"), ("date", "明日")],
         [("colleague", "他部署の担当者"), ("meeting", "会議"), ("date", "今週末")],
         [("colleague", "新しいお客様"), ("meeting", "商談"), ("date", "来月頭")]],
    20: [[("friend", "友人"), ("gift", "マグカップ")],
         [("friend", "妹"), ("gift", "アクセサリー")],
         [("friend", "同僚"), ("gift", "お菓子の詰め合わせ")],
         [("friend", "母"), ("gift", "花束")]],
    21: [[("task", "担当業務"), ("colleague2", "後任の方"), ("date2", "今週末")],
         [("task", "顧客対応"), ("colleague2", "新しいメンバー"), ("date2", "来週")],
         [("task", "経理業務"), ("colleague2", "異動してきた人"), ("date2", "月末")]],
    22: [[("friend2", "学生時代の友人"), ("meal", "食事"), ("place", "駅前のお店")],
         [("friend2", "元同僚"), ("meal", "ランチ"), ("place", "新しくできたカフェ")],
         [("friend2", "親戚"), ("meal", "夕食"), ("place", "行きつけの店")]],
    23: [[("service", "ジムの会員"), ("reason", "最近通えていない")],
         [("service", "英会話のオンライン講座"), ("reason", "時間が取れていない")],
         [("service", "新聞の定期購読"), ("reason", "あまり読めていない")]],
    24: [[("place", "カフェ"), ("color", "青")],
         [("place", "電車"), ("color", "黒")],
         [("place", "会社"), ("color", "透明")],
         [("place", "図書館"), ("color", "赤")]],
    25: [[("plant2", "トマト"), ("pest", "アブラムシ"), ("action2", "薬をまく")],
         [("plant2", "バラ"), ("pest", "ハダニ"), ("action2", "葉を洗う")],
         [("plant2", "きゅうり"), ("pest", "ナメクジ"), ("action2", "苗の周りを片付ける")],
         [("plant2", "レモンの木"), ("pest", "カイガラムシ"), ("action2", "枝を剪定する")]],
    26: [[("pet3", "犬"), ("toy", "お気に入りのボール"), ("place2", "庭")],
         [("pet3", "猫"), ("toy", "ねずみのおもちゃ"), ("place2", "ソファの下")],
         [("pet3", "犬"), ("toy", "ぬいぐるみ"), ("place2", "公園")],
         [("pet3", "うさぎ"), ("toy", "かじり木"), ("place2", "ケージの周り")]],
    27: [[("device", "スマホ"), ("repair_time", "3日後")],
         [("device", "パソコン"), ("repair_time", "1週間後")],
         [("device", "腕時計"), ("repair_time", "来月頭")],
         [("device", "洗濯機"), ("repair_time", "明日")]],
    28: [[("company_a", "A社"), ("company_b", "B社"), ("deadline2", "今週末")],
         [("company_a", "今の会社"), ("company_b", "新しい会社"), ("deadline2", "来週月曜")],
         [("company_a", "都心の会社"), ("company_b", "地元の会社"), ("deadline2", "今月末")]],
    29: [[("item2", "新しいカバン"), ("arrival_date", "明後日")],
         [("item2", "注文した本"), ("arrival_date", "来週")],
         [("item2", "誕生日プレゼント"), ("arrival_date", "今週金曜")],
         [("item2", "家電"), ("arrival_date", "明日の午前中")]],
    30: [[("person", "友人"), ("event_date", "来週土曜")],
         [("person", "母"), ("event_date", "来月の誕生日")],
         [("person", "同僚"), ("event_date", "退職する日")],
         [("person", "パートナー"), ("event_date", "結婚記念日")]],
    31: [[("furniture2", "ソファ"), ("price_from", "50000"), ("price_to", "40000")],
         [("furniture2", "食器棚"), ("price_from", "30000"), ("price_to", "25000")],
         [("furniture2", "デスク"), ("price_from", "20000"), ("price_to", "15000")]],
}

for idx, topic in enumerate(topics):
    for variant_slots in slot_variants[idx]:
        t = dict(topic)
        t["slots"] = variant_slots
        # 各バリエーションごとにcallback表現を変えて2〜3つ生成
        for _ in range(random.choice([2, 3])):
            blocks.append(build_dialogue(t))


def main() -> None:
    random.shuffle(blocks)
    text = "\n\n".join(blocks) + "\n"
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"書き出し: {OUT_PATH}")
    print(f"  会話数: {len(blocks)}")
    print(f"  文字数: {len(text):,}")


if __name__ == "__main__":
    main()
