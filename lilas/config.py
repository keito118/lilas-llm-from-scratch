from dataclasses import dataclass


@dataclass
class LilasConfig:
    # トークナイザ
    vocab_size: int = 1400

    # モデル構造
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128  # 一度に見られるトークン数(文脈長)
    dropout: float = 0.1

    # 学習
    steps: int = 7000
    batch_size: int = 32
    lr: float = 3e-4
    eval_interval: int = 300
    eval_iters: int = 20
    warmup_steps: int = 100

    # 生成
    max_new_tokens: int = 150
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.15
