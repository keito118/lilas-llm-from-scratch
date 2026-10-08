# Lilas

**English** | [日本語](README.ja.md)

Lilas is a small Japanese conversational language model (GPT-style, 40,241,152 parameters) built from scratch in PyTorch. No pretrained weights are used: the tokenizer, model, training loop, fine-tuning data and evaluation set are all part of this repository. Lilas speaks in the polite style of a secretary or assistant.

Sample from the frozen held-out set (`eval/results_round2_2026-09-10.md`, the current model, unedited):

```
User: やあ、久しぶり
Lilas: お久しぶりです!最近どうされていましたか?
User: 元気にしてた?
Lilas: はい、元気にしております。あなたはいかがですか?
```

```
User: ところで130円のジュースを3本買ったら合計いくら?
Lilas: 130円 × 3 = 390円です。合計390円になります。以上になります。
```

The same file also contains the model's weak answers (generic replies, missed callbacks), which are kept on purpose. See [Results](#results).

## Highlights

- **From scratch**: custom BPE tokenizer (with an incremental merge update that made training 28× faster) and a decoder-only Transformer, with top-p sampling and a repetition penalty that also applies to the previous turn.
- **Two-stage training**: pretraining on Aozora Bunko and a Wikipedia-ja sample, then fine-tuning on dolly-15k-ja, alpaca-gpt4-japanese and hand-written conversations that give Lilas her persona.
- **Context extension**: the first model (v1) was extended from 256 to 1024 tokens without retraining from scratch by copying the learned position embeddings and adapting (`scripts/extend_context.py`). The current model (v2) was pretrained at 1024 tokens directly.
- **Tool use**: rule-based calculator, Japanese public holidays, JMA weather, Wikipedia lookup and a two-item comparison tool, using only public APIs that need no key. Tool results are passed to the model as reference text (`(参考情報: ...)`) and the model writes the answer itself.
- **Measured progress**: a frozen held-out conversation set (`eval/held_out_conversations.md`, 12 conversations, plus a 6-conversation v2 set) evaluated after every training round. Results are kept in `eval/` whether the round was adopted or not.
- **Honest log**: `training_log.md` and `SCALE_UP_PLAN.md` record what went wrong, how it was diagnosed and what was decided.

## Architecture

| Item | v1 (2026-09-07/08) | v2, current (2026-09-09 onwards) |
|---|---|---|
| Model | Decoder-only Transformer | Decoder-only Transformer |
| Parameters | 6,852,608 | 40,241,152 |
| Layers / heads / embedding | 6 / 8 / 256 | 10 / 8 / 512 |
| Vocabulary (BPE, trained on the corpus) | 8,000 | 16,000 |
| Context length | 256 → 1024 (extended) | 1024 |
| Pretraining corpus | 178,298,814 tokens | Aozora Bunko + Wikipedia-ja sample |

```
corpus (Aozora Bunko + Wikipedia-ja sample)
        │  pretrain (lilas/train.py)
        ▼
   base model ──► (v1 only) extend context 256→1024 (scripts/extend_context.py)
        │  fine-tune on instruction + persona data (--curated-only)
        ▼
   chat model ──► lilas/chat.py
        │           ├─ lilas/calculator.py  (arithmetic)
        │           └─ lilas/tools.py       (holidays, weather, Wikipedia, comparison)
        ▼
   eval/run_eval.py  (frozen held-out conversations)
```

## Project structure

```
lilas/                  model code
  config.py             default hyperparameters
  tokenizer.py          BPE tokenizer
  model.py              Transformer, top-p sampling, repetition penalty
  data.py               corpus loading, persona/knowledge/greeting oversampling
  train.py              pretraining and fine-tuning
  chat.py               chat CLI, tool routing
  calculator.py         rule-based arithmetic
  tools.py              holidays, JMA weather, Wikipedia, comparison
scripts/
  fetch_datasets.py     download Aozora / Wikipedia / dolly / alpaca into index/
  extend_context.py     extend block_size of a checkpoint
  generate_*.py         template-based training data (seed, greetings, emotion,
                        grounded, knowledge_broad, longcontext, arithmetic)
eval/
  run_eval.py           run the frozen held-out set against a checkpoint
  held_out_conversations*.md   the frozen test sets (never used for training)
  results_*.md          output of each round
  parallel_test_*.md    stress-test reports by category
index/                  self-written training data (persona, seed, generated, Claude-authored)
training_log.md         troubleshooting log
SCALE_UP_PLAN.md        v2 scale-up plan and round-by-round results
skills.md               catalogue of skills Lilas should have
```

## Quick start

Python 3.11 was used. With an NVIDIA GPU, install a CUDA build of PyTorch first; `train.py` and `chat.py` use the GPU automatically when `torch.cuda.is_available()` is true.

```bash
pip install -r requirements.txt

# 1. Download external datasets into index/ (not included in this repo)
python scripts/fetch_datasets.py all        # Aozora Bunko + Wikipedia-ja sample (pretraining)
python scripts/fetch_datasets.py instruct   # dolly-15k-ja + alpaca-gpt4-japanese (fine-tuning)

# 2. Pretrain (v2 settings)
python -m lilas.train --vocab-size 16000 --n-embd 512 --n-layer 10 --n-head 8 \
  --block-size 1024 --batch-size 8 --steps 80000 --tag v2

# 3. (Optional) extend the context of an existing checkpoint, then adapt it
python scripts/extend_context.py checkpoints/lilas_pretrained_base.pt \
  checkpoints/lilas_extended_base.pt --new-block-size 1024
python -m lilas.train --init-from checkpoints/lilas_extended_base.pt \
  --steps 15000 --batch-size 16 --lr 0.00015 --tag context_adapted

# 4. Fine-tune for conversation
python -m lilas.train --init-from checkpoints/lilas_v2.pt --curated-only \
  --persona-repeat 50 --steps 8000 --batch-size 8 --lr 0.0001 --tag v2_ft

# 5. Chat (uses checkpoints/lilas.pt by default; --tag NAME uses checkpoints/lilas_NAME.pt)
python -m lilas.chat
python -m lilas.chat --once "こんにちは"

# 6. Evaluate on the frozen held-out set
python eval/run_eval.py --tag v2_ft --out eval/results_v2_ft.md
```

The current model (`round2`) was fine-tuned from `lilas_v2.pt` with `--persona-repeat 21` and 15,000 steps. `chat.py` loads `checkpoints/lilas.pt`, so copy or rename the checkpoint you want to use.

**Model weights and external datasets are not included** because of their size and licenses. `scripts/fetch_datasets.py` downloads the datasets.
<!-- TODO: if weights are published on Hugging Face or a GitHub Release, link them here -->

## Results

The held-out files contain conversations, not scores, so each round was judged by reading every conversation and comparing it with the current model. Numbers below are the ones recorded in `SCALE_UP_PLAN.md` and `training_log.md`.

| Round | What changed | Held-out result | Adopted |
|---|---|---|---|
| baseline (v1) | 6.85M model, 256→1024 context | Single-question test 16/17, but multi-turn conversations broke down (same wrong reply repeated 3 times) | — |
| v2_ft | Scaled up to 40M params, pretrained at 1024 | Broken sentences mostly gone; Mt. Fuji height (3,776 m) correct | Yes |
| v2_ft repfix | Repetition penalty covers the previous turn (decoding only) | Turn-to-turn repeats fixed in conversations 2, 4, 6, 7 | Yes |
| v2_ft2 | +8 hand-written multi-turn examples, 2nd fine-tune | val loss unchanged (2.7927) but new ungrammatical sentences | No |
| v2_ft3 | Templated callback + arithmetic data, from clean v2 | First generalised callback on an unseen name; arithmetic was memorised (130×3 right, 90×7 wrong) | No |
| calculator | Rule-based calculator instead of learning arithmetic | Arithmetic question in conversation 11 answered correctly | Yes |
| v2_ft4 | Phrase-level paraphrasing of templates | Callback held; template phrases leaked into unrelated turns | No |
| v2_ft5 | Pools of whole sentence structures instead of phrases | Template leaks gone, callback 3 runs in a row; Mt. Fuji regressed | No |
| v2_ft6 | 24 topics, 3× knowledge oversampling | Callback 4 in a row; Mt. Fuji still wrong → move facts to tools | No |
| v2_ft8 | Tools + reference-text (RAG-style) training, 714 examples | Calculator / holidays / time reliable; callback 7 in a row | Yes |
| v2_ft9 | Reference texts made to look like real Wikipedia output | Tokyo Tower, Mt. Fuji, Oda Nobunaga, Lake Biwa stable; callback 8 in a row | Yes |
| round1 | Greetings, longer callbacks, emotion data, consistency data | Greetings better; an unknown person was answered with another person's facts | No |
| **round2** | Removed the consistency block, more long-conversation focus examples | Cleanest held-out run; greetings appropriate in 16/16 samples | **Yes (current)** |
| round3 | 97 long-conversation focus examples, comparison tool | Long-conversation focus 2/2; comparison conclusion wrong 2/2; Everest regression | No |
| round4 | Comparison pairs that do not overlap single facts | Everest fixed; focus gain and person mix-up came back | No |
| round5 | Same as round4, different seed | Focus gain not reproduced → round3 gain was seed luck | No |
| round6 | 70 real Wikipedia entities for knowledge | New failure: fused names of two trained people | No |
| round7 | Only the 70 real entities | Entity fusion recurred in other forms | No |
| round8 | More implicit-emotion examples | Did not generalise to close paraphrases | No |

Other measurements on the current model: emotion inference improved from 30–40% to 66.7% in a stress test, and the knowledge tool answered 8 of 24 questions correctly (33%) when it fired.

## What I learned

- **val loss is not conversation quality.** Several rounds had the same or better val loss and were clearly worse when read; the best-val step was often less natural than the final step. Every round is now judged on the frozen held-out set.
- **Check "improvements" on unseen inputs.** 130 yen × 3 looked solved, but 90 × 7 came back as 300 yen. The answer was memorised, so arithmetic moved to a rule-based calculator, and later facts and comparisons moved to tools.
- **Algorithm cost shows up only at scale.** The first BPE trainer rescanned the whole corpus on every merge; updating only the affected words made it 28× faster (356 s → 12.65 s on the same test).
- **Inference-time tricks must be re-tested after every model change.** A self-introduction "priming" prefix that was harmless at 256 tokens made the 1024-token model introduce itself in reply to unrelated questions, and was removed.
- **Some limits are structural.** The tokenizer splits numbers irregularly (`50000` → `500`+`00`, `45000` → `45`+`000`), so recalling arbitrary numbers from context could not be learned; with ~40M parameters, trained entities kept fusing. In both cases the decision was to stop adding data and let code decide the answer.

Write-up: [Why my from-scratch Japanese LLM says "That sounds tough" to good news](https://dev.to/keito118/why-my-from-scratch-japanese-llm-says-that-sounds-tough-to-good-news-3k15) (dev.to). How counting the training data, then weighting it by the real training mix, traced why Lilas answered good news with sympathy.

## Data and licenses

| Data | License | Included |
|---|---|---|
| Aozora Bunko (`globis-university/aozorabunko-clean`) | Public domain (per work) | No, downloaded by script |
| Wikipedia-ja sample (`wikimedia/wikipedia`, 20231101.ja) | CC BY-SA | No, downloaded by script |
| dolly-15k-ja (`kunishou/databricks-dolly-15k-ja`) | CC BY-SA 3.0 | No, downloaded by script |
| alpaca-gpt4-japanese (`FreedomIntelligence/alpaca-gpt4-japanese`) | CC BY-NC 4.0 (GPT-4 generated) | No, downloaded by script |
| Self-written data in `index/` | Same as this repository | Yes |

## License

MIT License, see [LICENSE](LICENSE). Datasets keep their own licenses (see above).
