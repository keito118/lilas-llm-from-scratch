"""学習用コーパスの読み込みとバッチ生成"""

from __future__ import annotations

import array
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "index"

# Lilas固有の人格・自己紹介・手書き会話データ。instruct_*(公開の指示応答
# データセット)や external_*(事前学習用の大規模コーパス)と違い、
# 量は少ないがLilasの個性そのものを表すデータなので、ファインチューニング時に
# 複製して比率を上げられるようにしている(load_corpusのpersona_repeat参照)。
CORE_PERSONA_FILES = {
    "persona_lilas.md",
    "about_lilas_ja.txt",
    "seed_conversations_ja.txt",
    "generated_conversations_ja.txt",
    "claude_authored_conversations_ja.txt",
    "claude_authored_multiturn_ja.txt",
    "claude_authored_longcontext_ja.txt",
    "claude_authored_knowledge_ja.txt",
    "claude_authored_general_ja.txt",
    "claude_authored_longcontext2_ja.txt",
    "generated_longcontext_ja.txt",
    "claude_authored_naturalflow_ja.txt",
    "claude_authored_callback_ambiguous_ja.txt",
    "generated_grounded_ja.txt",
    "generated_greetings_ja.txt",
    "generated_emotion_ja.txt",
    "generated_knowledge_broad_ja.txt",
}
# 2026-09-09: v2_ft5の評価で、CORE_PERSONA_FILES全体に同じ倍率(persona_repeat)を
# かけると、生成したテンプレート会話(量が多い)に対して知識データ(量が少ない)が
# 相対的に薄まり、既知の正解(富士山の標高など)を誤答するようになる退行が
# 見つかった。知識・自己紹介系のファイルはさらに追加で複製し、量の多い
# テンプレート会話に埋もれないようにする。
KNOWLEDGE_BOOST_FILES = {
    "persona_lilas.md",
    "about_lilas_ja.txt",
    "claude_authored_knowledge_ja.txt",
    "claude_authored_general_ja.txt",
}
KNOWLEDGE_BOOST_MULTIPLIER = 3
# 2026-09-10: 並列テストで「こんにちは」等の挨拶語は、対話ファイル内では
# 5〜9回しか「挨拶→挨拶の返し」として登場しないのに対し、Wikipedia/
# instructコーパスでは2000回以上、無関係な文脈(用件に答え始める等)で
# 登場しており、比率が実に400:1以上にまで偏っていた。この偏りは
# KNOWLEDGE_BOOST_MULTIPLIER(3倍)では到底追いつかないため、
# generated_greetings_ja.txtには別枠でより大きい倍率をかける
GREETING_BOOST_FILES = {"generated_greetings_ja.txt"}
GREETING_BOOST_MULTIPLIER = 5
# 2026-09-09: generated_arithmetic_ja.txt は学習させても未知の数値に
# 汎化しない(丸暗記止まり)ことが確認できたため、CORE_PERSONA_FILESから
# 除外した(training_log.md参照)。計算はlilas/calculator.pyの
# ルールベース電卓で確実に処理する方針に切り替えたため、モデルの
# 学習コーパスに含める意味がない。ファイル自体は参考として残す。


def load_corpus(include_external: bool = True, persona_repeat: int = 1) -> str:
    """index/ 以下の全テキストファイルを連結してコーパスを作る

    注意: skills.md(開発ログ)はここに含めない。
    2026-09-07に試したところ、skills.mdの生の開発ログ(バグの説明や
    「もぐらたたき」のような比喩表現)がそのままコーパスの26%を占め、
    無関係な質問に対する応答としてそのまま漏れ出す問題が起きた。
    「Lilasに自分自身のことを学習させる」目的は、index/about_lilas_ja.txt
    のような、会話向けに書いた短い自己紹介文で果たすようにしている。

    include_external=False にすると、`external_`で始まるファイル
    (青空文庫やWikipediaなど、外部から取得した大規模コーパス)を除外し、
    手作りの会話データだけを対象にする。大規模コーパスで事前学習した後、
    対話フォーマットをファインチューニングする用途で使う
    (詳細はtraining_log.md参照)。

    persona_repeat: CORE_PERSONA_FILES を何回繰り返してコーパスに含めるか。
    2026-09-07に、公開のinstruction tuningデータ(dolly-15k-ja等、
    2300万文字)を追加したところ、Lilas固有の人格・自己紹介データ(2.4万文字、
    全体の0.1%)がほぼ埋もれてしまい、「君の名前は?」等の受け答えが
    崩れる問題が起きた。persona_repeatを上げて複製することで、
    Lilasらしさのデータの存在感を意図的に増やせる(詳細はtraining_log.md)。
    """
    texts: list[str] = []

    if INDEX_DIR.exists():
        for path in sorted(INDEX_DIR.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".txt", ".md"):
                continue
            if path.name == "README.md":
                continue
            if not include_external and path.name.startswith("external_"):
                continue
            content = path.read_text(encoding="utf-8")
            repeat = persona_repeat if path.name in CORE_PERSONA_FILES else 1
            if path.name in KNOWLEDGE_BOOST_FILES:
                repeat *= KNOWLEDGE_BOOST_MULTIPLIER
            if path.name in GREETING_BOOST_FILES:
                repeat *= GREETING_BOOST_MULTIPLIER
            texts.extend([content] * repeat)

    if not texts:
        raise FileNotFoundError(
            "学習データが見つかりません。index/ 以下にテキストファイルを置いてください。"
        )

    return "\n\n".join(texts)


def sample_for_tokenizer(corpus: str, max_chars: int, n_segments: int = 20) -> str:
    """トークナイザ学習用に、コーパス全体からまんべんなくサンプリングする。

    コーパスが大きくなると素朴なBPE学習は非現実的に遅くなるため、
    代表的な一部だけを使って語彙を学習する。コーパスの先頭だけを使うと
    特定のデータソースに偏るので、全体をn_segments個の区間に分けて
    各区間から少しずつ取り出す。
    """
    if len(corpus) <= max_chars:
        return corpus

    seg_len = len(corpus) // n_segments
    take_per_seg = max_chars // n_segments
    parts = [corpus[i * seg_len : i * seg_len + take_per_seg] for i in range(n_segments)]
    return "".join(parts)


def train_val_split(ids, val_ratio: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    # idsがarray.array('I')の場合、torch.tensor(ids, ...)に直接渡すと
    # 内部でPython intを1個ずつ取り出す遅いパス+巨大な中間listを
    # 作ってしまいメモリを圧迫する。numpy経由(バッファプロトコルで
    # ゼロコピー変換)にすることでこれを避ける。
    if isinstance(ids, array.array):
        np_arr = np.frombuffer(ids, dtype=np.uint32).astype(np.int64)
        data = torch.from_numpy(np_arr)
    else:
        data = torch.tensor(ids, dtype=torch.long)
    n = int(len(data) * (1 - val_ratio))
    return data[:n], data[n:]


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError(
            "コーパスが短すぎて block_size 分のバッチを作れません。"
            "index/ のデータを増やすか block_size を小さくしてください。"
        )
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
