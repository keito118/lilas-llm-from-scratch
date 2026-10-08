"""Lilasのトークナイザとモデルを学習するスクリプト

使い方:
    python -m lilas.train
    python -m lilas.train --steps 3000 --vocab-size 800 --n-layer 4 --n-embd 128
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from lilas.config import LilasConfig
from lilas.data import get_batch, load_corpus, sample_for_tokenizer, train_val_split
from lilas.model import LilasGPT
from lilas.tokenizer import BPETokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"


def parse_args() -> argparse.Namespace:
    cfg = LilasConfig()
    p = argparse.ArgumentParser(description="Lilasの学習")
    p.add_argument("--steps", type=int, default=cfg.steps)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--vocab-size", type=int, default=cfg.vocab_size)
    p.add_argument("--n-layer", type=int, default=cfg.n_layer)
    p.add_argument("--n-head", type=int, default=cfg.n_head)
    p.add_argument("--n-embd", type=int, default=cfg.n_embd)
    p.add_argument("--block-size", type=int, default=cfg.block_size)
    p.add_argument("--dropout", type=float, default=cfg.dropout)
    p.add_argument("--eval-interval", type=int, default=cfg.eval_interval)
    p.add_argument(
        "--retrain-tokenizer",
        action="store_true",
        help="既存のトークナイザがあっても再学習する",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="乱数シード。同じデータ・同じ設定でも毎回結果が変わるのを防ぐため固定する",
    )
    p.add_argument(
        "--tokenizer-sample-chars",
        type=int,
        default=3_000_000,
        help="トークナイザ学習に使う文字数の上限。コーパスがこれより大きい場合は"
        "全体からまんべんなくサンプリングする(素朴なBPE実装が巨大コーパスでは"
        "非現実的に遅いため)",
    )
    p.add_argument(
        "--curated-only",
        action="store_true",
        help="index/ の external_*(青空文庫やWikipediaなど外部の大規模コーパス)を"
        "除外し、手作りの会話データだけを学習対象にする。"
        "大規模コーパスで事前学習した後、対話フォーマットをファインチューニングする用途",
    )
    p.add_argument(
        "--persona-repeat",
        type=int,
        default=1,
        help="Lilas固有の人格・自己紹介データ(CORE_PERSONA_FILES)を何回複製して"
        "コーパスに含めるか。公開のinstruction tuningデータ等、量が多い他のデータに"
        "埋もれてLilasらしさが薄れるのを防ぐために使う",
    )
    p.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="このチェックポイント(.pt)の重みを初期値として読み込み、追加学習"
        "(ファインチューニング)する。指定した場合、モデルの構造(vocab_size等)は"
        "チェックポイントの設定に合わせ、--n-layer等の構造系オプションは無視される。"
        "トークナイザも既存のものをそのまま使う(vocab_sizeを変えると埋め込みが"
        "壊れるため)",
    )
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="保存するチェックポイントのファイル名に付けるタグ。"
        "指定すると lilas_<tag>.pt / lilas_<tag>_best_val.pt に保存される"
        "(ベースの事前学習モデルを上書きしないようにする用途)",
    )
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="混合精度学習(bf16)を無効にする。デフォルトはGPU+bf16対応環境で自動有効",
    )
    return p.parse_args()


@torch.no_grad()
def estimate_loss(
    model: LilasGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    cfg: LilasConfig,
    device: str,
    amp_ctx,
) -> dict[str, float]:
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.block_size, cfg.batch_size, device)
            with amp_ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    suffix = f"_{args.tag}" if args.tag else ""
    model_path = CHECKPOINT_DIR / f"lilas{suffix}.pt"
    best_val_model_path = CHECKPOINT_DIR / f"lilas{suffix}_best_val.pt"

    init_ckpt = None
    if args.init_from:
        if args.retrain_tokenizer:
            raise SystemExit(
                "--init-from と --retrain-tokenizer は同時に使えません"
                "(ファインチューニングは既存モデルと同じvocab_sizeである必要があるため)"
            )
        init_ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        # ファインチューニング時はモデル構造(vocab_size等)を元チェックポイントに合わせる。
        # 学習まわりの設定(steps/batch_size/lr等)だけCLI引数で上書きする
        cfg = LilasConfig(**init_ckpt["config"])
        cfg.steps = args.steps
        cfg.batch_size = args.batch_size
        cfg.lr = args.lr
        cfg.eval_interval = args.eval_interval
        cfg.dropout = args.dropout
    else:
        cfg = LilasConfig(
            vocab_size=args.vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            block_size=args.block_size,
            dropout=args.dropout,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_interval=args.eval_interval,
        )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 混合精度学習(bf16): forward計算をbf16で行うことでVRAM使用量と計算時間を
    # 減らせる。bf16はfp16と違ってオーバーフローしにくく、GradScalerなしで
    # 安全に使える(Ampere世代以降のGPU、RTX 3060含む、で利用可能)。
    # 2026-09-08まで一度も使っていなかった最適化(SCALE_UP_PLAN.md参照)。
    use_amp = (
        not args.no_amp and device == "cuda" and torch.cuda.is_bf16_supported()
    )
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    print(f"混合精度学習(bf16): {'有効' if use_amp else '無効'}")

    if device == "cpu":
        # デフォルトだと物理コア数を使い切らないことがあるので、
        # 使えるコード数いっぱいまでスレッド数を上げてCPU学習を高速化する
        n_threads = os.cpu_count() or 4
        torch.set_num_threads(n_threads)
        print(f"device: cpu (threads={n_threads})")
    else:
        print(f"device: {device}")

    print("コーパスを読み込み中...")
    corpus = load_corpus(
        include_external=not args.curated_only, persona_repeat=args.persona_repeat
    )
    print(f"  コーパスサイズ: {len(corpus):,} 文字 (persona_repeat={args.persona_repeat})")

    if TOKENIZER_PATH.exists() and not args.retrain_tokenizer:
        print(f"既存のトークナイザを読み込み: {TOKENIZER_PATH}")
        tokenizer = BPETokenizer.load(TOKENIZER_PATH)
        cfg.vocab_size = tokenizer.vocab_size
    else:
        sample = sample_for_tokenizer(corpus, args.tokenizer_sample_chars)
        print(
            f"トークナイザを学習中 (vocab_size={cfg.vocab_size}, "
            f"学習サンプル{len(sample):,}文字/全体{len(corpus):,}文字)..."
        )
        tokenizer = BPETokenizer()
        tokenizer.train(sample, cfg.vocab_size, verbose=True)
        tokenizer.save(TOKENIZER_PATH)
        print(f"  保存: {TOKENIZER_PATH}")

    # curated-only(ファインチューニング用の小さいコーパス)と、外部コーパスを
    # 含むフルコーパスとでキャッシュファイルを分けておく。同じファイルを
    # 共有すると、切り替えるたびに大きい方のコーパス(エンコードに約18分)の
    # キャッシュが消えてしまうため
    ids_cache_path = (
        CHECKPOINT_DIR / f"corpus_ids_cache{'_curated' if args.curated_only else ''}.pt"
    )
    corpus_hash = hashlib.sha256(
        corpus.encode("utf-8") + str(tokenizer.vocab_size).encode()
    ).hexdigest()
    ids: list[int] | None = None
    if ids_cache_path.exists():
        cache = torch.load(ids_cache_path, weights_only=False)
        if cache.get("hash") == corpus_hash:
            ids = cache["ids"]
            print(f"  トークン化済みキャッシュを再利用: {ids_cache_path}")

    if ids is None:
        print("コーパスをエンコード中(大きいコーパスだと時間がかかります)...")
        t_encode = time.time()
        ids = tokenizer.encode_large(corpus)
        torch.save({"hash": corpus_hash, "ids": ids}, ids_cache_path)
        print(f"  エンコード完了: {time.time() - t_encode:.1f}s")

    print(f"  トークン数: {len(ids):,}")
    train_data, val_data = train_val_split(ids)

    model = LilasGPT(cfg).to(device)
    if init_ckpt is not None:
        model.load_state_dict(init_ckpt["model_state"])
        print(f"チェックポイントから重みを初期化しました: {args.init_from}")
    print(f"モデルパラメータ数: {model.num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        return cfg.lr

    print(f"学習開始: {cfg.steps} steps")
    t0 = time.time()
    best_val = float("inf")
    best_step = -1
    for step in range(cfg.steps):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)

        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        with amp_ctx:
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % cfg.eval_interval == 0 or step == cfg.steps - 1:
            losses = estimate_loss(model, train_data, val_data, cfg, device, amp_ctx)
            elapsed = time.time() - t0
            marker = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                best_step = step
                torch.save(
                    {"model_state": model.state_dict(), "config": cfg.__dict__},
                    best_val_model_path,
                )
                marker = f" (best val, saved to {best_val_model_path.name})"
            print(
                f"step {step:5d} | train loss {losses['train']:.4f} "
                f"| val loss {losses['val']:.4f} | {elapsed:.1f}s{marker}"
            )

    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, model_path)
    print(f"最終ステップのモデルを保存しました: {model_path}")
    print(
        f"参考: val lossが最良だったのは step {best_step} (val loss={best_val:.4f})、"
        f"{best_val_model_path} に保存済み。"
    )
    print(
        "注意: このデータ規模では、val lossが低い=会話として自然、とは限らない"
        "(学習不足でむしろ崩れた文章になることがある)。"
        f"chat.py はデフォルトで最終ステップのモデル({model_path.name})を使う。"
    )


if __name__ == "__main__":
    main()
