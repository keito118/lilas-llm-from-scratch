"""Lilasと会話するCLI

使い方:
    python -m lilas.chat                  # 対話モード
    python -m lilas.chat --once "こんにちは"  # 1回だけ発話させる(自動テスト用)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lilas.calculator import try_calculate
from lilas.config import LilasConfig
from lilas.model import LilasGPT
from lilas.tokenizer import BPETokenizer
from lilas.tools import try_tools

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"

STOP_MARKERS = ["\nUser:", "\nUser", "User:", "\nLilas:", "\n\n"]

# 2026-09-07に試した「会話の最初に自己紹介のやり取りを"呼び水"として
# 入れておく」仕組み(SYSTEM_PRIMING)は廃止した。
# block_size=1024+2026-09-08の本番ファインチューニング後のモデルで検証したところ、
# この呼び水があると「君の名前は?」「日本の首都はどこですか?」のような
# 全く関係ない質問にまで自己紹介の答えを返してしまう、深刻な副作用が
# 判明した(呼び水なしなら正しく答えられることを確認済み)。
# 詳細はtraining_log.md参照。
SYSTEM_PRIMING = ""


def load_model_and_tokenizer(
    model_path: Path | None = None,
) -> tuple[LilasGPT, BPETokenizer, LilasConfig, str]:
    model_path = model_path or (CHECKPOINT_DIR / "lilas.pt")
    if not model_path.exists() or not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"学習済みモデルが見つかりません({model_path})。"
            "先に `python -m lilas.train` を実行してください。"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = LilasConfig(**checkpoint["config"])
    model = LilasGPT(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, tokenizer, cfg, device


def truncate_history(history: str, max_turns: int | None) -> str:
    """historyを直近max_turns往復だけに切り詰める。

    2026-09-08の自然な雑談テストで、履歴が伸びるほど直前の発言と
    無関係な定型文を引き当てる現象が見られた。履歴を直近数往復に
    絞ることで、モデルが「今何に答えるべきか」を見失いにくくなるか
    どうかを検証するための仕組み。max_turns=Noneなら切り詰めない
    (これまで通りの全履歴を使う挙動)。
    """
    if max_turns is None:
        return history
    blocks = [b for b in history.split("\n\n") if b.strip()]
    return "\n\n".join(blocks[-max_turns:]) + ("\n\n" if blocks else "")


def _extract_last_reply(history: str, tokenizer: BPETokenizer) -> tuple[str, list[int]]:
    """historyの一番最後のLilasの返答を(テキスト, トークンID列)で返す。

    2026-09-09の評価で判明した「前ターンとほぼ同じ返答をそのまま繰り返す」
    問題への対策。各ターンのgenerate()は独立した呼び出しなので、前ターンの
    返答はこのターンでは単なるプロンプトの一部になり、通常の
    repetition_penaltyでは再利用を抑制できない。直前の返答を明示的に
    抽出し、model.generate()のextra_penalize_idsとして渡す(かつ、
    それでも完全に同じ返答になった場合の再試行判定にも使う)。
    """
    blocks = [b for b in history.split("\n\n") if b.strip()]
    if not blocks:
        return "", []
    marker = "Lilas: "
    pos = blocks[-1].rfind(marker)
    if pos == -1:
        return "", []
    reply_text = blocks[-1][pos + len(marker):].strip()
    ids = tokenizer.encode(reply_text) if reply_text else []
    return reply_text, ids


def generate_reply(
    model: LilasGPT,
    tokenizer: BPETokenizer,
    cfg: LilasConfig,
    device: str,
    history: str,
    user_input: str,
    max_history_turns: int | None = None,
) -> tuple[str, str]:
    """historyとuser_inputから応答を1つ生成する。(応答テキスト, 更新後history) を返す"""
    # 学習データの書式("Lilas: "とコロンの後にスペース)に合わせる。
    # ここがズレると、モデルがスペースやタグ自体を生成し直してしまう。
    history = truncate_history(history, max_history_turns)

    # 2026-09-09(前半): 計算・事実知識・天気・祝日・現在時刻は、モデルに
    # 学習させても複製倍率を上げるだけでは汎化・定着しないことが分かった
    # (training_log.md参照)。当初はモデルを介さずツールの答えをそのまま
    # 返す方式にしていたが、開発者の方針転換により「調べた事実をLilasに
    # 見せて、Lilasの言葉で答えさせる」方式(RAG)に変更した。まだ推論を
    # 学習していないモデルなので最初はうまく使いこなせないかもしれないが、
    # 「育てる」過程として、多少ミスってもモデルに考えさせることを優先する。
    # 2026-09-10: 以前は「計算が見つかったら知識/天気/時刻は一切試さない」
    # 実装だったため、「天気を教えて、あと5+3は?」のような複合質問で
    # 片方が黙って握りつぶされる不具合があった。両方試して見つかった分を
    # 連結する
    facts = [f for f in (try_calculate(user_input), try_tools(user_input)) if f is not None]
    fact = "\n".join(facts) if facts else None

    prev_reply_text, prev_reply_ids = _extract_last_reply(history, tokenizer)
    # 生成用のプロンプトには参考情報を含めるが、historyに永続的に
    # 積み重なっていくと学習データに無い書式がターンを追うごとに増えて
    # 悪影響が出かねないので、履歴として保存する方は通常の書式のままにする
    plain_prompt = history + f"User: {user_input}\nLilas: "
    if fact is not None:
        prompt = history + f"User: {user_input}(参考情報: {fact})\nLilas: "
    else:
        prompt = plain_prompt
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    extra_penalize_ids = (
        torch.tensor(prev_reply_ids, dtype=torch.long, device=device)
        if prev_reply_ids
        else None
    )

    def _run(temperature: float, repetition_penalty: float) -> str:
        out = model.generate(
            idx,
            max_new_tokens=cfg.max_new_tokens,
            temperature=temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            repetition_penalty=repetition_penalty,
            extra_penalize_ids=extra_penalize_ids,
        )
        new_ids = out[0, len(ids):].tolist()
        text = tokenizer.decode(new_ids)
        for marker in STOP_MARKERS:
            pos = text.find(marker)
            if pos != -1:
                text = text[:pos]
        return text.strip()

    reply = _run(cfg.temperature, cfg.repetition_penalty)
    if prev_reply_text and reply == prev_reply_text:
        # extra_penalize_idsをかけても直前と一字一句同じ返答になった場合、
        # サンプリングを一度だけ強めて再試行する(2026-09-09の評価で
        # 確認された「前ターンの返答をそのまま繰り返す」現象への対策)
        reply = _run(
            min(cfg.temperature + 0.2, 1.5), cfg.repetition_penalty * 1.3
        )

    updated_history = plain_prompt + reply + "\n\n"
    return reply, updated_history


def main() -> None:
    p = argparse.ArgumentParser(description="Lilasと会話する")
    p.add_argument("--once", type=str, default=None, help="1回だけ発話させて終了する")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None, help="nucleus sampling (0-1)")
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="1より大きくすると同じ内容の繰り返しを抑える(1.0で無効)",
    )
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="読み込むチェックポイントのタグ(train.pyの--tagと対応)。"
        "指定すると checkpoints/lilas_<tag>.pt を読み込む",
    )
    p.add_argument(
        "--max-history-turns",
        type=int,
        default=None,
        help="会話履歴を直近何往復まで保持するか。未指定なら全履歴を使う"
        "(履歴が伸びるほど直前の発言と無関係な応答が増える問題への対策として試験導入)",
    )
    args = p.parse_args()

    model_path = CHECKPOINT_DIR / f"lilas_{args.tag}.pt" if args.tag else None
    model, tokenizer, cfg, device = load_model_and_tokenizer(model_path)
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.top_p is not None:
        cfg.top_p = args.top_p
    if args.repetition_penalty is not None:
        cfg.repetition_penalty = args.repetition_penalty

    if args.once is not None:
        reply, _ = generate_reply(
            model, tokenizer, cfg, device, SYSTEM_PRIMING, args.once, args.max_history_turns
        )
        print(reply)
        return

    print("Lilasと話せます。終了するには 'exit' か Ctrl+C。")
    history = SYSTEM_PRIMING
    while True:
        try:
            user_input = input("あなた: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nバイバイ!")
            break

        if user_input.lower() in ("exit", "quit"):
            print("バイバイ!")
            break
        if not user_input:
            continue

        reply, history = generate_reply(
            model, tokenizer, cfg, device, history, user_input, args.max_history_turns
        )
        print(f"Lilas: {reply}")


if __name__ == "__main__":
    main()
