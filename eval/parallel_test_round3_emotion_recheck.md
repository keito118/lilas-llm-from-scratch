# Round 2 Production Model Re-check — Emotional Inference & Ambiguous Input Handling

**Date:** 2026-09-10
**Model tested:** `checkpoints/lilas.pt` (confirmed = round2 checkpoint, currently in production)
**Method:** Fresh, non-overlapping test scenarios (none reused from `scripts/generate_emotion_data.py`'s `IMPLIED_NEGATIVE`, `IMPLIED_POSITIVE`, `AMBIGUOUS`, `AMBIGUOUS_MIDCONV`, `EMOTIONAL_PIVOT` blocks — new situations, same underlying skill). All runs read-only via `lilas.chat.generate_reply`.

## Summary scorecard

| Category | N | Pass | Hit rate |
|---|---|---|---|
| Implied-negative valence match | 12 | 8 | 66.7% |
| Implied-positive valence match | 6 | 4 | 66.7% |
| **Combined implied-valence (1+2)** | **18** | **12** | **66.7%** |
| Ambiguous/low-content → no hallucinated narrative | 8 | 8 | 100%* |
| Emotional pivot respected (doesn't dwell after "but I'll manage") | 3 | 2 | 66.7%† |

\* But 2/8 (25%) of these were incoherent/non-sequitur replies rather than hallucinations — a different, still-present quality problem.
† 1 full pass, 1 partial/garbled pass, 1 clear fail (dwelled on the negative after user resolved it).

**Estimated overall valence/grounding hit rate: ~65–70%**, vs. the ~30–40% baseline estimated pre-training. This is a real, roughly-doubling improvement — but still fails on about 1 in 3 cases, so it is not yet reliable enough to consider "fixed."

## Section 1: Implied-negative (facts only, no emotion words)

8/12 correct (matched negative valence). Failures:
- 「上司に出した資料、書き直しって突き返された」→ *"それはお役に立てて嬉しいです、ありがとうございます。"* — inverted to positive/gratitude.
- 「頼んでた荷物、住所を間違えて別の場所に届いてたみたい」→ *"それは待ち遠しいですね。"* — inverted to positive anticipation.
- 「予約してたレストラン、行ったら席が無いって言われた」→ *"それは楽しみですね、今度行ってみたいですね。"* — inverted to positive.
- 「誕生日なのに、家族が誰も覚えてなかったみたい」→ *"それは楽しみですね!どれくらい長く続けられたか教えてください。"* — inverted to positive + hallucinated follow-up question.

This is the same polarity-inversion failure mode reported originally (positive reaction to bad news), just less frequent than baseline.

## Section 2: Implied-positive

4/6 correct. Failures:
- 「苦手だった上司が異動になったんだ」→ *"それはご心配ですね。"* — inverted to negative/worry.
- 「無くしたと思ってた指輪、コートのポケットから出てきたよ」→ *"指輪、コートのポケットから出てきたよ"* — degenerate: echoed the user's own sentence back instead of generating a reply.

## Section 3: Ambiguous/low-content input

Tested as opener (4), and mid-conversation after a direct question (4, e.g. 「休みの日は何してるの?」→「いや、特には」). **All 8/8 avoided fabricating an unrelated specific story** — a clear, genuine improvement over the original 「びみょう」→ invented-snack-story failure. Typical replies: *"そうなんですね。もしお話ししたいことがあれば伺いますよ。"*

However, 2/8 replies were low-quality non-sequiturs rather than good acknowledgments (e.g. 「それなりに」→ *"はい、こちらこそどうぞよろしくお願いいたします。"*; 「別になんでもいいや」→ *"別に、何かあればいいですか?"*). Not hallucination, but not coherent either.

**Important new finding — hallucination has not disappeared, it has shifted:** while literal ambiguous filler words are now handled well, ordinary short direct questions/statements *not* resembling the trained ambiguous phrases still trigger fabricated, unrelated, oddly-specific narratives:
- 「今度の旅行、楽しみ?」→ *"楽しみにしていらっしゃった沖縄旅行ですね、その後の進み具合はいかがでしょうか?"* (invented "Okinawa trip" out of nothing)
- 「新しい職場はどう?」→ *"新しいメンバー様へのサプライズでしたね、来週土曜のご予定でしたか、準備はいかがですか?"* (invented an unrelated "surprise for new members next Saturday")
- 「昨日提出したレポート、間違いが見つかったんだ」→ *"それは心配ですね。新しい部署に異動になったらしいですでしょうか。"* (invented a "department transfer")

This suggests the round2 fix generalized narrowly to the specific trained ambiguous tokens (「びみょう」「なんか」etc.) rather than to the broader skill of "don't invent ungrounded specifics." The core hallucination tendency is still present and shows up whenever the model lacks strong grounding, regardless of whether the input was ambiguous by design.

## Section 4: Emotional pivot (negative → "but I'll manage")

- 「でも直せば済むことだし、切り替えていくよ」→ *"とても良い取り組みですね。"* — good, respects the pivot, doesn't dwell. **PASS**
- 「まあ、次から気をつければいいだけだから大丈夫」→ *"それは何よりです、気になります。"* — mixed: opens positive ("that's good") but tacks on an incongruous "気になります" (I'm concerned). **Partial**
- 「薬飲んだから、もう平気だと思う」→ *"それは大変そうですね、無理なさらないでくださいね。"* — ignores the resolution entirely and dwells on "sounds tough, don't overdo it." **FAIL**

Also observed mid-conversation, unrelated to the pivot turn itself: two more hallucinated-narrative instances (see Section 3) and one garbled/incoherent reply (「だるいし頭も痛いんだけど」→ *"だるいしは、毎朝くらいのペースでやってるよ"*, nonsensical).

## Overall assessment

- **Has it improved?** Yes, measurably. Implied-valence inference went from an estimated ~30–40% baseline to ~67% in this fresh test — roughly a doubling. The literal ambiguous-filler hallucination problem (「びみょう」→ fabricated snack story) that was the headline original failure appears genuinely fixed for inputs resembling the trained patterns.
- **Has it fully resolved the underlying issues?** No. Two failure patterns remain common enough to matter:
  1. **Polarity inversion** (~1 in 3 implied-emotion cases): reacting positively to clearly bad news or negatively to clearly good news, when no explicit emotion word is present.
  2. **Narrow generalization of the anti-hallucination fix**: grounded-narrative fabrication still occurs readily on ordinary direct questions/statements outside the specific trained ambiguous-filler vocabulary — the model still invents unrelated specific details (destinations, departments, dates) when it lacks strong signal from the input.
- A secondary, less severe issue also surfaced: outright **incoherent/non-sequitur replies** (garbled or echoing the user's own text back) in a handful of cases (~3 across ~35 total exchanges), independent of valence or hallucination.

**Recommendation:** round2 is a genuine step forward but should not be treated as "done" for this issue class. Promoting round3/4/5 (which reportedly contain further data changes since round2) and re-running this same fresh test suite against them would clarify whether later rounds close the remaining ~33% gap, particularly on polarity inversion and on hallucination outside the literal trained ambiguous phrases.
