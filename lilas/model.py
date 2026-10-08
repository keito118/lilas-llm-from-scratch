"""
GPT系デコーダのみのTransformerを自作実装したもの。

nn.MultiheadAttentionのような既製のAttention実装は使わず、
Attentionの計算そのものを自分たちで書いている。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from lilas.config import LilasConfig


class CausalSelfAttention(nn.Module):
    """未来のトークンを見ないようにマスクをかけたMulti-Head Self-Attention"""

    def __init__(self, cfg: LilasConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.qkv_proj = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)

        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v  # (B, nh, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


class MLP(nn.Module):
    def __init__(self, cfg: LilasConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return self.dropout(x)


class Block(nn.Module):
    """Pre-LayerNorm Transformerブロック: Attention + MLP、それぞれ残差接続あり"""

    def __init__(self, cfg: LilasConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LilasGPT(nn.Module):
    def __init__(self, cfg: LilasConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # 入力Embeddingと出力層の重みを共有(weight tying)
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, "block_size を超える系列長が渡されました"

        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        extra_penalize_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """次のトークンを1つずつサンプリングして生成する。

        top_k: 確率上位k個のトークンだけを候補にする
        top_p (nucleus sampling): 確率の高い順に足し合わせて、累積確率が
            top_pを超えるまでのトークンだけを候補にする。top_kと違い、
            分布がとがっている(自信がある)ときは候補が少数に絞られ、
            分布がなだらかな(自信がない)ときは候補が増える、という
            自然な絞り込みになる
        repetition_penalty: 1より大きくすると、「今回の応答内で」既に
            生成したトークンをもう一度選びにくくする(1.0で無効)。
            同じ文の繰り返しループを抑えるための簡易的な対策。
            プロンプト(それまでの会話履歴)側のトークンにはかけない
            (助詞など普通に再利用されるべきトークンまで避けてしまい、
            不自然な文になるため)
        extra_penalize_ids: プロンプト側のトークンだが、repetition_penaltyの
            対象に含めたいID列(1次元)。2026-09-09の評価で、対話の各ターンが
            独立したgenerate()呼び出しになるため「前ターンとほぼ同じ返答を
            そのまま繰り返す」現象が見つかった(前ターンの返答は次のターンでは
            プロンプト側になり、通常のrepetition_penaltyの対象外になるため)。
            直前のLilasの返答トークンをここに渡すことで、そのターン限定で
            軽く再利用しにくくする
        """
        prompt_len = idx.size(1)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    pieces = []
                    if idx.size(1) > prompt_len:
                        pieces.append(idx[b, prompt_len:])
                    if extra_penalize_ids is not None and extra_penalize_ids.numel() > 0:
                        pieces.append(extra_penalize_ids)
                    if not pieces:
                        continue
                    seen = torch.unique(torch.cat(pieces))
                    seen_logits = logits[b, seen]
                    logits[b, seen] = torch.where(
                        seen_logits > 0, seen_logits / repetition_penalty,
                        seen_logits * repetition_penalty,
                    )

            logits = logits / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                # 累積確率がtop_pを超えた最初のトークンより後ろを切り捨てる
                # (超えた瞬間のトークン自体は候補として残す)
                remove = cumulative - sorted_probs > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(
                    -1, sorted_idx, sorted_logits
                )

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
