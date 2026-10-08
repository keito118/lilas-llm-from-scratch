"""
既存のチェックポイントのblock_size(文脈長)を拡張するスクリプト。

ゼロから学習し直すのではなく、学習済みの重みのほとんど(トークン埋め込み・
Attention・MLP・LayerNorm)をそのまま引き継ぎ、位置埋め込み(pos_emb)だけを
拡張する。位置埋め込みは元のblock_size分だけ学習済みの値をコピーし、
延長した分は新規に初期化する。Attention/MLP/LayerNormの重みはトークンの
位置に依存しない演算なので、そのままのサイズで使い回せる。

拡張しただけでは延長した位置の使い方をモデルはまだ知らないので、
このあと必ず新しいblock_sizeで追加学習(継続事前学習)を行うこと。

使い方:
    python scripts/extend_context.py checkpoints/lilas_pretrained_base.pt checkpoints/lilas_extended_base.pt --new-block-size 1024
"""

from __future__ import annotations

import argparse

import torch

from lilas.config import LilasConfig
from lilas.model import LilasGPT


def extend_context(src_path: str, dst_path: str, new_block_size: int, seed: int = 1234) -> None:
    torch.manual_seed(seed)

    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    old_cfg = LilasConfig(**ckpt["config"])
    old_block_size = old_cfg.block_size

    if new_block_size <= old_block_size:
        raise ValueError(
            f"new_block_size({new_block_size})はold_block_size({old_block_size})より"
            "大きくしてください"
        )

    new_cfg = LilasConfig(**{**ckpt["config"], "block_size": new_block_size})
    new_model = LilasGPT(new_cfg)  # ランダム初期化(位置埋め込み・causal_maskも新サイズ)
    new_state = new_model.state_dict()

    old_state = ckpt["model_state"]
    for key, old_tensor in old_state.items():
        if key not in new_state:
            continue
        if key.endswith("causal_mask"):
            # block_sizeに依存するバッファ。新モデル構築時に正しいサイズで
            # 作られているので、古い方は使わずそのままにする
            continue
        if key == "pos_emb.weight":
            # 元のblock_size分だけ学習済みの値をコピーし、
            # 延長した分は新モデルのランダム初期化のままにする
            new_state[key][:old_block_size, :] = old_tensor
            continue
        new_state[key] = old_tensor

    new_model.load_state_dict(new_state)

    torch.save(
        {"model_state": new_model.state_dict(), "config": new_cfg.__dict__}, dst_path
    )
    print(f"block_size {old_block_size} -> {new_block_size} で保存しました: {dst_path}")
    print(
        "注意: 拡張しただけでは延長した位置の使い方をモデルは学習していません。"
        "必ず新しいblock_sizeで継続学習してください。"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="チェックポイントのblock_sizeを拡張する")
    p.add_argument("src", help="元のチェックポイントパス")
    p.add_argument("dst", help="保存先のチェックポイントパス")
    p.add_argument("--new-block-size", type=int, required=True)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    extend_context(args.src, args.dst, args.new_block_size, args.seed)


if __name__ == "__main__":
    main()
