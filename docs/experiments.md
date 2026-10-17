# Retrieval Experiments

Eval: 15 saved collections (size ≥ 9) as ground truth, query = collection name, k = 10
Corpus: 28,796 documents (corpus.jsonl) → 36,767 chunks (400 chars, overlap 50)

| Tag | Method | Index | Hit@10 | P@10 | R@10 | MRR | Note |
|---|---|---|---|---|---|---|---|
| v0 | vector, e5-small | all sources (36,767 chunks, 62 min) | 0.733 | 0.233 | 0.142 | 0.340 | baseline |
| E1_bge_m3 | vector, bge-m3 | saved only (2,061 chunks, 46.7 min) | 0.733 | 0.213 | 0.124 | 0.344 | no gain, ~5x slower → keep e5-small |
| E2_vector | vector, e5-small | all sources (insta_v0) | 0.733 | 0.233 | 0.142 | 0.340 | sanity check = v0 |
| E2_bm25 | BM25 (word + Korean 2-gram) | saved docs (1,123) | 0.733 | 0.220 | 0.157 | 0.474 | strong on abbreviations |
| E2_hybrid | RRF(vector, BM25), k=60 | — | 0.667 | 0.227 | 0.156 | 0.492 | best MRR, lost 인테리어 / IT Information |
| E3_w1.5 | weighted RRF (vector 1.5 : BM25 1) | — | 0.733 | 0.233 | 0.155 | 0.497 | **chosen** — no loss vs v0, MRR +46% |
| E3_w2 | weighted RRF (2 : 1) | — | 0.733 | 0.233 | 0.155 | 0.497 | same as 1.5 (stable plateau) |
| E3_w3 | weighted RRF (3 : 1) | — | 0.733 | 0.227 | 0.154 | 0.448 | too vector-heavy |
| E6_w1.5_rr | hybrid w1.5 + bge-reranker-v2-m3 (top 30) | — | 0.800 | 0.227 | 0.147 | 0.476 | ~55 s/query on CPU; not default |


## Findings
- E1: bigger model did not help → hypothesis C (weak model) rejected
- E1: full-index estimate with bge-m3 = 36,767 / 2,061 × 46.7 min ≈ 13.9 h (vs 62 min for e5-small) → about 13 h avoided
- Abstract collection names (인재, 배우고 싶은 것들) fail with every method → query side is the bottleneck
- E2: hybrid raised MRR 0.340 → 0.492 (first correct answer moves up ~1 rank on average)
- E2: equal-weight RRF lets BM25 noise push out vector hits → try weighted RRF (E3)

- E3: vector weight 1.5 recovers Hit (0.733) and keeps MRR 0.497 → default for ask.py
- Caution: weight tuned on the same 15 queries (overfitting risk); small because 1 parameter + plateau 1.5–2
- Still failing: 인테리어, 인재, 책 정보, 배우고 싶은 것들 → query expansion later

- E4: ask.py → hybrid (w1.5) + one result per post (fixes duplicate [2]/[4])
- E4: natural question "내가 저장한 의학 공부 관련 게시물 알려줘" was worse with hybrid (filler word "저장" matched "저장하고 보세요")
- E5: stopword removal before search ("… 알려줘" → "의학 공부") → medical posts 2–3/5 → 4/5; LLM still receives the original question
- Remaining noise: generic word "공부" pulls a language-study post → reranker later

- E6: reranker wins on topic-describing captions (Ml, 영어, CS, 인테리어 → RR 1.00) but loses on content collections (노래, 운동, Math) — it judges "is this ABOUT the word", while collections hold "items OF the category"
- E6: Hit 0.733 → 0.800 but MRR 0.497 → 0.476 and ~55 s/query → keep as optional --rerank, not default
- LLM (qwen2.5:3b): citations OK, but mis-summarized an irrelevant source and added an ungrounded closing sentence → compare a larger LLM in the cloud stage
- Caution: E1–E6 all tuned on the same 15 queries → build a larger natural-question eval set (50+) before further tuning