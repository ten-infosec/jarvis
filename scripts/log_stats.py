# ===== Summarize the request log (operations metrics) =====
# Usage: python scripts/log_stats.py


import json
from pathlib import Path


LOG_FILE = Path(r"C:\instagram_export\logs\requests.jsonl")


def percentile(values, p):
    # values must be sorted; p = 0.5 is the median, p = 0.95 is p95
    if not values:
        return 0
    index = round((len(values) - 1) * p)
    return values[index]


def summary(name, values):
    values = sorted(values)
    if not values:
        print(f"{name:10} (no data)")
        return
    avg = sum(values) / len(values)
    print(f"{name:10} n={len(values):3d}  avg={avg:6.2f}s  median={percentile(values, 0.5):6.2f}s  "
          f"p95={percentile(values, 0.95):6.2f}s  max={values[-1]:6.2f}s")


# ===== Load =====
rows = []
with open(LOG_FILE, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

ok = [r for r in rows if r.get("status") == "ok"]
errors = [r for r in rows if r.get("status") == "error"]

# ===== Overview =====
print(f"Requests: {len(rows)}   ok: {len(ok)}   errors: {len(errors)}   "
      f"error rate: {len(errors) / len(rows) * 100 if rows else 0:.1f}%")
# Lines are written when a request FINISHES, so file order != start order -> use earliest/latest
times = sorted(r["ts"] for r in rows)
print(f"Period:   {times[0] if times else '-'}  ~  {times[-1] if times else '-'}\n")
# ===== Latency =====
summary("retrieve", [r["retrieve_sec"] for r in ok])
# Generation time only counts requests that actually called the LLM
summary("generate", [r["generate_sec"] for r in ok if not r["retrieve_only"]])
summary("total", [r["total_sec"] for r in ok])

# ===== Slowest requests =====
print("\nSlowest 3:")
for r in sorted(ok, key=lambda r: -r["total_sec"])[:3]:
    print(f"  {r['total_sec']:6.2f}s  {r['ts']}  {r['question'][:40]}")

# ===== Recent errors =====
if errors:
    print("\nRecent errors:")
    for r in errors[-3:]:
        print(f"  {r['ts']}  {r['error'][:80]}")


# → requests.jsonl을 읽어서 전체 요청 수, 에러율, 단계별 응답 시간(평균, 중앙값, p95, 최대)을 출력하겠다.
#   가장 느렸던 질문 3개와 최근 에러도 보여 줘서 어디를 개선해야 할지 바로 알 수 있게 한다.