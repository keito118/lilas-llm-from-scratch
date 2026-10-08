"""
バイトレベルBPE (Byte Pair Encoding) トークナイザの自作実装。

テキストをまずUTF-8バイト列にする(0〜255の256トークンからスタート)。
そこから、コーパス中で最も頻繁に隣り合うトークンのペアを繰り返しマージしていき、
指定した語彙サイズになるまで新しいトークンを作っていく。

バイトレベルで扱うので、日本語のように単語間にスペースがない言語でも
事前の分かち書きなしにそのまま学習・エンコードできる。
"""

from __future__ import annotations

import array
import json
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm

# 文字種(ひらがな/カタカナ/漢字/英数字/空白)ごとのまとまりでテキストを
# 事前分割する正規表現。日本語には英語のような単語区切り(スペース)が
# ないので、文字種の切り替わりを簡易的な単語境界として使う。
# これにより、BPEのマージ探索を「コーパス全体」ではなく「数文字程度の
# 短い塊」単位で行えるようになり、同じ塊(助詞や頻出の単語)はキャッシュも
# 効くので、桁違いに大きいコーパスでも現実的な時間でエンコードできる。
_PRETOKENIZE_RE = re.compile(
    r"[ぁ-んー]+|[ァ-ヴー]+|[一-龠々]+|[a-zA-Z0-9]+|\s+|."
)


class BPETokenizer:
    def __init__(self) -> None:
        # token id -> そのトークンが表すバイト列
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # マージ操作の順序: (id1, id2) -> 新しいid。学習時に登場した順で保持する
        self.merges: dict[tuple[int, int], int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """語彙を学習する。

        単純な実装だと、1マージごとに「全ユニーク単語を再スキャンして
        ペア頻度を数え直す」ため、マージ回数×ユニーク単語数に比例して
        遅くなる(実測: vocab1200・300万文字で356秒。vocab8000だと
        50分近くかかる計算になり非現実的)。

        そこで差分更新にする: あるペアをマージしたとき、そのペアを
        含んでいた単語だけがペア頻度に影響するので、影響を受けた単語だけを
        更新し、無関係な単語は触らない。`pair_to_words`で「このペアを
        含む単語インデックス」を管理し、マージのたびに影響範囲だけ
        pair_counts を差分更新する。
        """
        if vocab_size < 256:
            raise ValueError("vocab_size は256以上にしてください(バイト分だけで256必要)")

        piece_counts = Counter(_PRETOKENIZE_RE.findall(text))
        # 各ユニークな「まとまり」を [バイトid列, 出現回数] として保持する
        words: list[list] = [
            [list(piece.encode("utf-8")), freq] for piece, freq in piece_counts.items()
        ]

        pair_counts: Counter = Counter()
        pair_to_words: dict[tuple[int, int], set[int]] = {}

        def add_word_pairs(wi: int) -> None:
            ids, freq = words[wi]
            for a, b in zip(ids, ids[1:]):
                pair_counts[(a, b)] += freq
                pair_to_words.setdefault((a, b), set()).add(wi)

        def remove_word_pairs(wi: int) -> None:
            ids, freq = words[wi]
            for a, b in zip(ids, ids[1:]):
                pair_counts[(a, b)] -= freq

        for wi in range(len(words)):
            add_word_pairs(wi)

        num_merges = vocab_size - 256
        for i in range(num_merges):
            # 頻度0以下(過去の差分更新のカス)を無視して最大のペアを探す
            best_pair = None
            best_count = 0
            for pair, count in pair_counts.items():
                if count > best_count:
                    best_count = count
                    best_pair = pair
            if best_pair is None or best_count <= 0:
                break

            new_id = 256 + i
            affected = pair_to_words.pop(best_pair, set())
            for wi in affected:
                remove_word_pairs(wi)
                words[wi][0] = self._merge(words[wi][0], best_pair, new_id)
                add_word_pairs(wi)
            pair_counts.pop(best_pair, None)

            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose and (i + 1) % 50 == 0:
                print(f"  merge {i + 1}/{num_merges}: {best_pair} -> {new_id} "
                      f"({self.vocab[new_id]!r})")

    @staticmethod
    def _count_pairs(ids: list[int]) -> Counter:
        counts: Counter = Counter()
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] += 1
        return counts

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        if not self.merges:
            return ids

        while len(ids) >= 2:
            pairs = self._count_pairs(ids)
            # 学習時にマージした順番が早いペアを優先してマージする
            candidate = min(
                (p for p in pairs if p in self.merges),
                key=lambda p: self.merges[p],
                default=None,
            )
            if candidate is None:
                break
            ids = self._merge(ids, candidate, self.merges[candidate])
        return ids

    def encode_large(self, text: str, cache_size: int = 500_000) -> list[int]:
        """大きなテキスト向けのencode。

        素朴にencode()をテキスト全体に対して1回呼ぶと、マージ探索が
        テキスト全体の長さに比例して繰り返され(1回のマージ適用ごとに
        配列全体を再スキャンする実装のため)、数百MB規模のテキストでは
        現実的な時間で終わらない(実測: 200万文字で2分以上)。

        ここでは `_PRETOKENIZE_RE` で文字種のまとまり(数文字程度の
        「疑似単語」)に事前分割し、まとまり単位でencode()を呼ぶことで
        1回あたりの探索範囲を大幅に縮める。さらに、同じまとまり
        (助詞や頻出の単語)は何度も出現するので、結果をキャッシュして
        2回目以降はO(1)で済ませる。自然言語には強い偏り(Zipf分布)が
        あるため、この2つだけで数百倍〜規模によっては桁違いに高速化できる。
        """
        # 2026-09-09: corpus全体に対して一度にfindall()すると、ピース(疑似単語)
        # 一つ一つがPythonの文字列オブジェクトとしてすべてメモリ上に並ぶことになり、
        # 数億文字規模のコーパス(9.57億文字)ではMemoryErrorになった
        # (システムRAM 16GB程度に対して、findallの結果リストだけで
        # 数十GB規模になっていたと考えられる)。テキストを一定サイズの
        # チャンクに分けて、チャンクごとにfindall→encode→破棄することで、
        # メモリ上に同時に保持するピース数を抑える。
        # idsはPythonのlistではなくarray.array('I')(符号なし4バイト整数)で
        # 保持する。数億トークン規模になると、Pythonのintオブジェクトの
        # list(要素あたり28バイト以上)ではメモリが足りなくなる
        # (実測: 16GB RAM環境で9.57億文字のコーパスがMemoryErrorになった)。
        # array.arrayならC言語の配列と同じ密な表現(要素あたり4バイト)になり、
        # 7倍以上メモリ効率が良い。
        cache: dict[str, list[int]] = {}
        ids = array.array("I")
        total_chars = len(text)
        chunk_chars = 20_000_000  # 一度にfindallするテキスト量の上限

        with tqdm(total=total_chars, desc="encoding", unit="char", unit_scale=True) as pbar:
            for chunk_start in range(0, total_chars, chunk_chars):
                chunk = text[chunk_start : chunk_start + chunk_chars]
                pieces = _PRETOKENIZE_RE.findall(chunk)
                for piece in pieces:
                    cached = cache.get(piece)
                    if cached is None:
                        cached = self.encode(piece)
                        if len(cache) < cache_size:
                            cache[piece] = cached
                    ids.extend(cached)
                pbar.update(len(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "vocab_size": self.vocab_size,
            # merges は (id1, id2) -> new_id。JSONはtupleキー不可なので文字列化する
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()],
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        tok = cls()
        for a, b, new_id in data["merges"]:
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        return tok
