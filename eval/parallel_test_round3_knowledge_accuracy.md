# Round 3: Why correctly-injected facts still produce wrong answers

Goal: given that `try_knowledge()` fetches a correct fact from Wikipedia ~100%
of the time, why is the model's *final answer* only right ~33% of the time
(8/24 in the earlier stress test)? Tested 15 fresh questions (5 trained-entity,
10 untrained-entity, mixing "の高さ"/"って誰"/"とは" question types) plus 3
questions re-sampled 3x each (9 more generations), all against the production
checkpoint `checkpoints/lilas.pt`, read-only, via `try_knowledge()` +
`generate_reply()`.

## Raw results

### Trained entities (appear in `scripts/generate_grounded_data.py`'s `FACTS` list)

| Q | Fact len | Correct? | Notes |
|---|---|---|---|
| 富士山の高さは? | 125 | **NO** (run1) | Fact has "3775.56 m" and "3776.12m" (two close numbers). Reply: "はい、333.127" — a number that appears **nowhere** in the fact (looks like bleed-through from the 東京タワー=333m trained fact tested moments earlier). |
| エベレストの高さは? | 173 | **NO** | Fact: "...1954年にインド測量局が周辺12か所で測定し...8848 mという数値が..." Reply just repeats the intro sentence and states no number at all — the real number (8848) is buried after two decoy numbers (1954, 12). |
| 東京タワーの高さは? | 132 | **YES** | "Wikipediaによると、333メートルとのことでした。" Correct despite decoy numbers (1958, 12, 23) earlier in the fact. |
| 織田信長って誰? | 79 | **YES** | Reply almost verbatim reproduces this entity's own trained FACTS entry. |
| 桜とは? | 121 | **NO (partial)** | Fact says 落葉広葉樹 (deciduous); reply says 常緑広葉樹 (evergreen) — flipped the classifying word — plus appends an unrelated hallucinated sentence. |

Trained hit rate: **2/5 clearly correct (40%)**.

### Untrained entities (not in FACTS, pure real-time Wikipedia injection)

| Q | Fact len | Correct? | Notes |
|---|---|---|---|
| スカイツリーの高さは? | 154 | **YES** | "スカイツリーの高さは634 mです。" |
| 大阪城の高さは? | 170 | **NO** | Tool's own detail-regex grabbed an irrelevant sentence ("本丸と詰の丸の地表差は3メートル以上ある"), and the model's reply ignored it entirely, answering "江戸時代の活動は戦国時代から江戸時代初期だそうです" — that exact phrase is the trained FACTS entry for **徳川家康**, not 大阪城. |
| マッターホルンの高さは? | 136 | **YES** | Number (4,478m) sits early in the fact, right after the German label — reply correct. |
| 阿蘇山の高さは? | 111 | **NO** | Fact clearly states "標高1,592メートル" (single, unambiguous number). Reply: "はい、3,776メートルです。" — **exactly Fuji's trained height**, hallucinated in place of the injected number. |
| 名古屋城の高さは? | 128 | **NO** | Wikipedia extract had no numeric height in range searched (detail extraction empty). Reply invents "愛知県中部地方を拠点とする超高層ビルである" — that "超高層ビル" phrasing is lifted verbatim from the trained **あべのハルカス** fact. |
| 福沢諭吉って誰? | 141 | **NO** | Fact correctly says 幕末〜明治期. Reply says "幕末から大正時代にかけて活動した、薩長同盟の一員である" — "薩長同盟" is **坂本龍馬's** trained fact, not Fukuzawa's, and the era is wrong (大正 instead of 明治). |
| 大谷翔平って誰? | 141 | **NO** (tool-level failure) | Wikipedia search for "大谷翔平" actually returned the **Los Angeles Times** newspaper article (retrieval bug, not generation bug). Model faithfully summarized the (wrong) fact it was given. |
| 猫とは? | 147 | **NO** | Fact mentions "イヌ（犬）と並ぶ代表的なペット". Reply calls the cat "犬" (dog) — swapped ネコ/イヌ. |
| ラーメンとは? | 270 (longest tested) | **YES (mostly)** | Long, slightly redundant, but faithfully paraphrases the ingredients/soup base with no factual error. |
| ChatGPTとは? | 151 | **NO** | Reply: "はい、2023年11月時点です。" — ignores the definition entirely and answers in the date-stamp format `try_datetime()` uses, not `try_knowledge()`'s "起源/定義" format. |

Untrained hit rate: **2/10 clearly correct + 1 partial (ラーメン) ≈ 20–30%**.

### Repeated sampling (same question, 3 fresh generations, temperature 0.8)

| Q | Results | Interpretation |
|---|---|---|
| 富士山の高さは? | correct (3,776m), correct (3,776m), correct (3,776m) | All 3 repeats correct, even though the very first test of this exact question (above) gave the wildly wrong "333.127". **Same question flips between correct and badly wrong purely from sampling** — this one is largely random. |
| 猫とは? | wrong (猫→犬), wrong (猫→犬), wrong (猫→犬) | All 3 samples make the *same* ネコ/イヌ substitution. This is a **deterministic, structural** failure — re-sampling will not fix it; the model reliably conflates the two words whenever they co-occur in the source text. |
| スカイツリーの高さは? | correct ("634 mです"), correct-ish ("634 mphです" – garbled unit but right number), wrong ("2021年5月22日です" – answered with a date instead of the height) | Mixed — 2/3 usably correct, 1/3 wrong. Partial randomness. |

**Conclusion on hypothesis 5**: failure is a mix of both. Some questions (富士山, スカイツリー) are genuinely coin-flip sensitive to sampling — a plain retry would likely fix them a meaningful fraction of the time. Others (猫) fail the same specific way on every sample — no amount of re-sampling at inference time would fix these; they need a different fact shape or more training coverage of that failure pattern.

## Testing the four structural hypotheses

**1. Fact length.** No meaningful correlation. Mean length of facts that produced a correct answer ≈ 154 chars; mean length of facts that produced a wrong answer ≈ 143 chars — essentially the same, and the *single longest* fact tested (ラーメン, 270 chars) produced one of the *better* answers, while several under-130-char facts (阿蘇山 111, 名古屋城 128, 桜 121) failed. Length is not the driver.

**2. Position of the key number.** Weak/secondary effect. A single, unambiguous number with no other numbers nearby in the fact (マッターホルン, スカイツリー) tends to succeed. But it's not simply "early = good": 東京タワー's number came after three decoy numbers (1958/12/23) and still succeeded, while エベレスト's number came after two decoys (1954/12) and failed — and 阿蘇山's number was the *only* number in the whole fact, unambiguous and not particularly buried, yet still got overridden by a hallucinated trained value (3,776m). Multiple-candidate-number facts do seem to hurt, but a clean single number is no guarantee of success either.

**3. Trained vs. untrained entities — the dominant, most specific finding.** Trained-entity questions succeeded 2/5 (40%); untrained-entity questions succeeded ~2–3/10 (20–30%). The raw rate gap alone is modest given the small sample, but the *qualitative* pattern is unambiguous and repeats across nearly every untrained-entity failure: **the model doesn't fail by going blank or ignoring the fact — it fails by substituting a specific, verbatim fragment memorized from the 27-entity `FACTS` training set**, triggered purely by matching the question's *template shape* (e.g. any "Xの高さは?" question), regardless of which entity X actually is:
   - 阿蘇山の高さ → **3,776m** (memorized 富士山 number)
   - 大阪城の高さ → **"戦国時代から江戸時代初期"** (memorized 徳川家康 phrase)
   - 名古屋城の高さ → **"超高層ビル"** (memorized あべのハルカス classification)
   - 福沢諭吉って誰 → **"薩長同盟"** (memorized 坂本龍馬 phrase)

   This happened in 4 of the 5 clearly-wrong untrained answers. It is the single most consistent signature in the entire test. The model learned "when a question looks like *this shape*, answer with *this memorized content*" from the narrow, only-27-entity training set, and that learned shortcut is currently stronger than the (much less-trained) skill of reading a novel number/fact out of the freshly-injected context. This is exactly "memorization dressed up as RAG": the injected real-time fact is present in the prompt, but the model's strongest available pattern is still "recall the training example that matches this template," not "extract from context."

**4. Question type.** Modest effect, in the expected direction: height questions (with the dedicated numeric-detail-extraction pattern) got 3/8 correct (37.5%); "who" questions got 1/3 (33%); general "what/とは" definition questions (no numeric extraction, just a copied intro sentence) got 1/4 clearly correct (25%, one more partial). Real but secondary compared to finding #3 — and note ChatGPTとは's failure wasn't even a definition-quality problem, it was the model answering in the *wrong tool's output format* entirely (date-stamp style), suggesting some "とは" failures are about the model not reliably recognizing which templated response-shape a `Wikipediaで調べたところ...について:` fact should map to.

## Primary conclusion

The dominant driver of failure is **not** fact length and only secondarily position-of-number or question-type. It is that **the ~40M-parameter model has learned the 27-entity training set as memorized template-matched answers rather than as examples of the general skill "read the injected fact and report what it says."** When the entity is one of the 27 trained facts and the question phrasing is close to what was trained, the model does reasonably (2/5, and would likely do better with matching phrasing/paraphrase templates). When the entity is novel, the model still recognizes "this question shape wants a number/phrase back," but instead of reading the *new* number out of the *current* prompt, it reproduces the specific memorized number/phrase from whichever trained example the question shape most resembles — a hallucination that is highly confident and highly specific, not random noise. This is corroborated by the repeated-sampling test: some failures are genuinely random (fixable by resampling), but the cross-contamination failures (阿蘇山→3776m, 大阪城→徳川家康's phrase) are the more alarming, template-driven kind, and 猫→犬 shows a deterministic in-context confusion that resampling never fixes.

A secondary, separate failure mode was also found: the retrieval layer itself can hand the model a wrong fact (大谷翔平 → Los Angeles Times article) or an empty/irrelevant "detail" sentence (大阪城, 名古屋城) when the numeric-detail regex doesn't find a clean match near the top of the article — in these cases the model can't be expected to answer correctly no matter how good its RAG-reading skill is, since it was never given the right information.

## Recommendation

The highest-leverage fix is **not** shortening facts or reordering where the number sits — it's diversifying the *training signal that teaches the RAG-reading skill itself*, decoupled from any specific entity's content:
1. **Massively broaden the entity coverage in `generate_grounded_data.py`'s `FACTS`/knowledge-block generation**, ideally auto-generating many hundreds of distinct (topic, fact-string) pairs — possibly by sampling real Wikipedia extracts for many random topics at data-generation time, the same way `try_knowledge()` does at inference time — rather than the current fixed hand-written list of ~27 entities. The goal is to break the "template shape → specific memorized entity answer" shortcut by making sure no single number/phrase is reliably correlated with a given question template.
2. Include **negative/robustness examples** in training: same question template, deliberately different injected facts (including facts with decoy numbers, and facts about *different* topics of the same "shape", e.g. multiple different mountains/buildings/people), so the model is forced to actually condition on the specific fact text rather than the topic-independent template.
3. Separately, fix the retrieval-layer issues surfaced here (大阪城/名古屋城 empty detail extraction, 大谷翔平 wrong Wikipedia search hit) since those are gating the ceiling on RAG accuracy independent of the model's reading behavior.
4. Fact length and number position are not worth optimizing first — the data here doesn't support them as the primary lever, though keeping facts to a single unambiguous number where possible is a reasonable low-cost secondary cleanup.
