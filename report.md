# NimbusPay ticket triage: report

**Model:** `Qwen/Qwen2.5-3B-Instruct` with LoRA r=16 on all 7 projections; the base is loaded in fp16, not 4-bit.
**Final result on dev (200 rows, `score.py`):** 94.0% exact match, 98.8% mean field accuracy, 100% valid JSON.
The same base model scores 0% zero-shot.

## 1. Data audit (Part A)

I read tickets before training anything. Most problems surfaced when I wrote each SCHEMA.md rule as a function, ran
it over every row and read the rows that failed. Of 1,445 rows, I kept 1,262: 183 dropped and 175 fixed (205 field
fixes), all in `cleaning_log.csv`. `notebooks/data_prep.ipynb` reproduces that file byte for byte.

**Dropped:**
- **25 truncated labels.** They are 63-137 characters, against 147-194 for valid labels, and all stop before
  `language` and `needs_human`. I dropped them rather than guess the missing fields.
- **48 dev tickets.** 37 matched exactly. 11 more differed only by whitespace, "Thx" or a greeting, found by
  character n-gram cosine similarity. I set the 0.83 threshold after reading the 0.78-0.85 band by hand
  (`figures/fig4_train_dev_similarity.png`).
- **100 duplicates** (60 exact, 40 near-exact with identical labels) and **10 placeholder tickets** ("test",
  "asdf asdf", one empty) that still carried full labels.

**Fixed:**
- **70 legacy `prio: low/medium/high` keys.** The mapping matches the v2 rules on all 70, so this is a renamed key,
  not a different meaning.
- **30 DD/MM/YYYY dates** (17 have a day above 12 and none a month above 12, so DD/MM is certain) and **20
  un-normalised transaction ids**.
- **25 amounts stored in rupees, not paise.** Some had also lost their decimals (₹594.50 stored as `594`). I
  re-derived each from the ticket, only where exactly one mention could match.
- **60 priorities that broke the v2 rules.** `needs_human` settled who was wrong. On all 15 rows where the rule
  says P1 and the label P3, `needs_human` is true, as for real P1 tickets. Some disagreements came from the rupee
  bug, so I recomputed priority only after every input to the rule was fixed.

**Checked and unchanged:** category, channel, txn_id, txn_date, needs_human and language. On category, a
cross-validated TF-IDF classifier (97% accurate) gave a second opinion but found nothing worth overruling. Train
also settled three unwritten conventions: text in quoted earlier emails doesn't count, an old ticket number is not
"contacted before", and a scratch-card reward is not a payment channel.

## 2. Pipeline and experiments (Parts B-D)

**Setup.** The T4 has no bf16, so training uses fp16 autocast with a GradScaler and fp32 LoRA weights. The loss
covers only the answer and `<|im_end|>` (29.8% of tokens). `max_len` is 2,304, since the longest ticket has 2,215
tokens, so nothing is truncated; length-grouped batches raise padding efficiency from 58.7% to 80.2%. AdamW at
lr 2e-4, 5% warm-up, cosine decay, effective batch 16, 3 epochs, a resumable checkpoint every 25 steps. The loop
is hand-written (no TRL), so I control exactly what is masked.

Each run changes one thing against R1. With 200 dev rows one ticket is 0.5 points, so I compare runs with a paired
McNemar test on the same rows.

| Run | Change vs R1 | Exact | Field | Peak GiB | Min | Fixed/broke vs R1 |
|---|---|---|---|---|---|---|
| B1 | Base model, zero-shot | 0.0% | 0.0% | 7.7 | - | |
| B2 | Base + rules in prompt | 23.5% | 80.6% | 8.9 | - | |
| **R1** | Clean data, fp16, r16 | 91.5% | 98.2% | 10.4\* | 27.5 | |
| D1 | Raw data | 92.5% | 98.5% | 10.0 | 30.1 | 5/3, p=0.73 |
| D2 | 4-bit QLoRA | 93.5% | 98.7% | 8.3 | 34.3 | 8/4, p=0.39 |
| D3 | Rank 8 | 90.0% | 98.1% | 9.7 | 27.0 | 2/5, p=0.45 |
| D4 | + augmentation | 95.0% | 99.0% | 10.0 | 38.1 | 8/1, **p=0.039** |
| **Final** | D4, retrained | **94.0%** | **98.8%** | 10.0 | 34.1 | 8/3, p=0.23 |

B1/B2 peaks are for inference. \*R1 ran before I fixed a bug that kept the previous model in GPU memory; runs of
the same shape peak at 9.96 GiB.

- **Baselines.** Zero-shot, the model wraps its answer in code fences and invents its own keys. With the rules in
  the prompt it produces valid JSON, but gets priority right only 51% of the time and copies SCHEMA.md's example id
  `NP1234567890` into 13 answers.
- **Raw vs clean (D1).** This is the most important result. D1 looked better, but it scores 100% on the 47 dev
  tickets it saw in training. On the 153 leak-free rows it ties R1 (90.2% vs 89.5%, p=1.0). Cleaning did not buy
  accuracy; it made the dev number honest.
- **4-bit (D2) and rank 8 (D3)** are both within noise of R1. 4-bit saves 1.7 GiB but is 25% slower; with memory to
  spare, I kept fp16 and r16.
- **Augmentation (D4).** Most of R1's priority errors sat on the ₹5k/₹50k thresholds or a legal threat, where train
  has few examples. I added 370 clean tickets with one change each: 250 with an amount moved near a threshold (in
  the ticket's own style, e.g. `47.5k`) and 120 with a legal-threat sentence from train, relabelled by the rules.
  It was the only change that beat R1 significantly.

**Caveat on the final number.** The D4 configuration, trained three times, scored 95.0%, 93.5% and 94.0%: GPU
non-determinism alone is worth about a point. All three beat R1, but only one significantly, so I call the gain
real but small (about +2.5 points). The submitted model is the notebook's own retrain: 20/20 regenerated test
predictions match the file, and batched and one-at-a-time decoding agree on 40/40 dev rows.

## 3. Error analysis (Part E)

The final model gets 12 of 200 dev tickets wrong. By field: priority 7, needs_human 5, category 3, txn_date 3,
txn_id 1. Refund is the weakest category (87.3% exact), then payment_failed (92.5%). Amount, channel and language
are perfect.

- **Thresholds (dv-00037, -00098, -00127).** The amount is read correctly but compared wrongly: "284669 rupees"
  (₹2.8 lakh, no commas) is not treated as ≥ ₹50k, and `41.5k` and ₹4,836 are each pushed up one level.
- **Distractors and shortcuts (dv-00180, -00185).** One priority follows the ₹2,41,670 account *balance* instead of
  the ₹3,130 claim. Another ticket (₹28,904) becomes P1, apparently because it opens with "URGENT!!".
- **Category (dv-00010, -00177, -00157).** "I didn't see the money I added" and "a red flag on my top up" become
  fraud; the second label is debatable. Typo-heavy Hinglish fraud ("OTP manag ke mere piase nikal liye") becomes
  account_access, pulled by the word OTP.
- **Dates (dv-00069, -00147, -00120).** Two tickets give no date and the model invents one, once even *after*
  Received. *parso* ("day before yesterday") is resolved as five days back.
- **txn_id (dv-00124).** An id right after "Pichla ticket number #13657" is missed.

**The pattern:** the model reads values well but sometimes applies the rules wrongly. Every priority error has a
correct amount, and all 5 needs_human errors follow from a wrong P1 decision. A likely reason is key order: the
output puts `priority` second, so the model commits to it *before* writing the amount that decides it.

**Test set, without labels.** Test contains phrasings that appear in neither train nor dev: order numbers (26
tickets), receipt references (33), "this/last month" (20), "the sum involved" (15), convenience fees (12) and
spaced "N P" ids (12). The predictions show the gap. 20 transaction ids are order or receipt numbers (`58038`,
`PAY10079`), although SCHEMA.md says order numbers are not transaction ids. One channel is `googlepay` (not an
allowed value), and one date falls after Received. The spec covered these cases, but train never showed them. I chose **not** to tune on what I saw in test, since
that would turn the test score into a measure of my peeking rather than of the model.

## 4. AI tools I used

I used Claude (Anthropic, through Claude Code) as a pair programmer and tutor: to plan the audit and experiments
and challenge my reasoning, to draft notebook code that I ran on Colab and checked against the outputs, and to
assemble the final notebook and draft the README and this report, which I edited. Cleaning decisions rest on code
output and tickets I read myself, and several findings (the leak-free split, the three-run spread) came from
questioning results that looked too good. No AI model produced any label or prediction.
