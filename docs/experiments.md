# Retrieval Experiments

Eval: 15 saved collections (size ≥ 9) as ground truth, query = collection name, k = 10

| Tag | Method | Index | Hit@10 | P@10 | R@10 | MRR | Note |
|---|---|---|---|---|---|---|---|
| v0 | vector, e5-small | all sources (36,767 chunks, 62 min) | 0.733 | 0.233 | 0.142 | 0.340 | baseline |
| E1_bge_m3 | vector, bge-m3 | saved only (2,061 chunks, 46.7 min) | 0.733 | 0.213 | 0.124 | 0.344 | no gain, ~5x slower → keep e5-small |
| E2_vector | vector, e5-small | all sources (insta_v0) | 0.733 | 0.233 | 0.142 | 0.340 | sanity check = v0 |
| E2_bm25 | BM25 (word + Korean 2-gram) | saved docs (1,123) | 0.733 | 0.220 | 0.157 | 0.474 | strong on abbreviations |
| E2_hybrid | RRF(vector, BM25), k=60 | — | 0.667 | 0.227 | 0.156 | 0.492 | best MRR, lost 인테리어 / IT Information |

## Findings
- E1: bigger model did not help → hypothesis C (weak model) rejected
- Abstract collection names (인재, 배우고 싶은 것들) fail with every method → query side is the bottleneck
- E2: hybrid raised MRR 0.340 → 0.492 (first correct answer moves up ~1 rank on average)
- E2: equal-weight RRF lets BM25 noise push out vector hits → try weighted RRF (E3)