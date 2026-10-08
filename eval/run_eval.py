"""
凍結テストセット(held_out_conversations.md)を使って、チェックポイントの
会話品質を確認するスクリプト。このファイルが読むテストデータは
絶対に学習には使わない(index/には置かない)。

使い方:
    python eval/run_eval.py                          # デフォルトのlilas.pt
    python eval/run_eval.py --tag naturalflow         # checkpoints/lilas_naturalflow.pt
    python eval/run_eval.py --out eval/results_v6.md  # 結果をファイルに保存
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from lilas.chat import CHECKPOINT_DIR, generate_reply, load_model_and_tokenizer

EVAL_DIR = Path(__file__).resolve().parent
CONVERSATIONS_PATH = EVAL_DIR / "held_out_conversations.md"


def parse_conversations(path: Path) -> list[tuple[str, list[str]]]:
    """held_out_conversations.md を (見出し, User発言のリスト) のリストにパースする"""
    text = path.read_text(encoding="utf-8")
    conversations = []
    current_title = None
    current_turns: list[str] = []

    for line in text.splitlines():
        heading = re.match(r"^## (.+)$", line)
        if heading:
            if current_title is not None and current_turns:
                conversations.append((current_title, current_turns))
            current_title = heading.group(1)
            current_turns = []
            continue
        user_line = re.match(r"^User: (.+)$", line)
        if user_line and current_title is not None:
            current_turns.append(user_line.group(1))

    if current_title is not None and current_turns:
        conversations.append((current_title, current_turns))
    return conversations


def main() -> None:
    p = argparse.ArgumentParser(description="凍結テストセットで会話品質を評価する")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--out", type=str, default=None, help="結果を書き出すMarkdownファイル")
    p.add_argument("--max-history-turns", type=int, default=None)
    p.add_argument(
        "--conversations",
        type=str,
        default=None,
        help="使う凍結テストセットのパス(未指定ならheld_out_conversations.md)。"
        "追加の凍結セット(例: held_out_conversations_v2.md)を指定できる",
    )
    args = p.parse_args()

    model_path = CHECKPOINT_DIR / f"lilas_{args.tag}.pt" if args.tag else None
    model, tokenizer, cfg, device = load_model_and_tokenizer(model_path)

    conversations_path = Path(args.conversations) if args.conversations else CONVERSATIONS_PATH
    conversations = parse_conversations(conversations_path)
    print(f"{len(conversations)}件の凍結テスト会話を実行します\n")

    # 結果ファイルにローカルの絶対パスを残さないよう、プロジェクトルートからの相対パスで書く
    checkpoint_label = (
        model_path.relative_to(CHECKPOINT_DIR.parent).as_posix() if model_path else "lilas.pt"
    )
    output_lines = [f"# 評価結果 (checkpoint: {checkpoint_label})\n"]

    for title, turns in conversations:
        print(f"## {title}")
        output_lines.append(f"## {title}\n")
        history = ""
        for u in turns:
            reply, history = generate_reply(
                model, tokenizer, cfg, device, history, u, args.max_history_turns
            )
            print(f"User: {u}")
            print(f"Lilas: {reply}")
            output_lines.append(f"User: {u}\n")
            output_lines.append(f"Lilas: {reply}\n")
        print()
        output_lines.append("\n")

    if args.out:
        out_path = Path(args.out)
        out_path.write_text("".join(output_lines), encoding="utf-8")
        print(f"結果を保存しました: {out_path}")


if __name__ == "__main__":
    main()
