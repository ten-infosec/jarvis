# Cloud Deployment (GCP Cloud Run + Gemini)

- Service URL: https://jarvis-rag-121185169470.asia-northeast3.run.app
- Project: jarvis-rag-511912 / Region: asia-northeast3 (Seoul)
- Data: public arXiv metadata only (no private data in the cloud image)

## Deploy (one command)

```powershell
.\scripts\deploy.ps1 -Version 0.4 -DryRun   # print the commands only
.\scripts\deploy.ps1 -Version 0.4           # tests -> build -> push -> deploy -> smoke check
```
The script stops at the first failed step, refuses uncommitted changes and reused version tags,
and passes every argument separately (no comma/quote mistakes). Env vars, secrets and limits are kept.

### Manual fallback

```powershell
docker build -f Dockerfile.cloud -t jarvis-cloud:<ver> .
docker tag jarvis-cloud:<ver> asia-northeast3-docker.pkg.dev/jarvis-rag-511912/jarvis/jarvis-cloud:<ver>
docker push asia-northeast3-docker.pkg.dev/jarvis-rag-511912/jarvis/jarvis-cloud:<ver>
gcloud run deploy jarvis-rag --image asia-northeast3-docker.pkg.dev/jarvis-rag-511912/jarvis/jarvis-cloud:<ver> --region asia-northeast3
```

## After each deploy
- `/health` answers, `/docs` returns 404 (the script checks both)
- Ask one question, then check `jsonPayload.xff_hops` in the log (expected 1 on Cloud Run).
  If it is not 1, set TRUSTED_PROXY_HOPS so the rate limit uses the real client IP.
- Rollback: `gcloud run services update-traffic jarvis-rag --region asia-northeast3 --to-revisions <previous-revision>=100`

## Settings
- Runtime service account: jarvis-run (only roles/secretmanager.secretAccessor on gemini-api-key)
- GEMINI_API_KEY from Secret Manager (never in image or command line)
- min-instances 0 (cost 0 when idle), max-instances 1 (cost ceiling)
- App limits: 5 req/min per IP, 200 req/day, question 2-500 chars, top_n <= 10
- Client IP = rightmost X-Forwarded-For entry (minus TRUSTED_PROXY_HOPS, default 0); tracked-IP table is bounded
- Cloud mode hides /docs and internal error details (details go to the log only)

## Cost safety
- Gemini key lives in a project WITHOUT billing (free tier only, 429 instead of charges)
- Budget "jarvis-rag-cap": Cloud Run spend cap 8,000 KRW/month (pauses service)
- Budget "jarvis-rag-alerts": all services, email at 50/90/100% of 10,000 KRW

## Incidents
- 2026-10-17 Embedding 400 "unexpected model name format":
  PowerShell turned the unquoted comma in --set-env-vars into a space, so two variables were stored as one.
  Fix: quote the value -> --update-env-vars "A=1,B=2". Service stayed up thanks to BM25 fallback.
- 2026-10-17 Gemini embedding free daily quota hit at 900/2,000 papers -> resume after quota reset.

- 2026-10-17 App request log was written to /tmp only -> lost on scale-to-zero.
  Fix: also print one-line JSON to stdout (structured logging), searchable with gcloud logging read.

## Logs
```powershell
# Plain text logs (startup, HTTP access)
gcloud run services logs read jarvis-rag --region asia-northeast3 --limit 20

# Structured app logs (JSON on stdout -> Cloud Logging jsonPayload; kept even when instances scale to 0)
gcloud logging read "resource.type=cloud_run_revision AND jsonPayload.message:ask" --limit 20 --format="value(timestamp,jsonPayload.message,jsonPayload.total_sec,jsonPayload.question)"

# Errors only
gcloud logging read "resource.type=cloud_run_revision AND jsonPayload.status=error" --limit 20
```

## Tests & CI
```powershell
pip install -r requirements-dev.txt
python -m pytest tests -v      # 25 tests, fake engines: no API key, no network
```
GitHub Actions (`.github/workflows/tests.yml`) runs the tests on every push and blocks tracked `.env`,
data files (data/, *.jsonl, *.csv, chroma) and API-key-like strings.
