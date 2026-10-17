# Cloud Deployment (GCP Cloud Run + Gemini)

- Service URL: https://jarvis-rag-121185169470.asia-northeast3.run.app
- Project: jarvis-rag-511912 / Region: asia-northeast3 (Seoul)
- Data: public arXiv metadata only (no private data in the cloud image)

## Build & push
docker build -f Dockerfile.cloud -t jarvis-cloud:<ver> .
docker tag jarvis-cloud:<ver> asia-northeast3-docker.pkg.dev/jarvis-rag-511912/jarvis/jarvis-cloud:<ver>
docker push asia-northeast3-docker.pkg.dev/jarvis-rag-511912/jarvis/jarvis-cloud:<ver>

## Deploy (image only; other settings are kept)
gcloud run deploy jarvis-rag --image asia-northeast3-docker.pkg.dev/jarvis-rag-511912/jarvis/jarvis-cloud:<ver> --region asia-northeast3

## Settings
- Runtime service account: jarvis-run (only roles/secretmanager.secretAccessor on gemini-api-key)
- GEMINI_API_KEY from Secret Manager (never in image or command line)
- min-instances 0 (cost 0 when idle), max-instances 1 (cost ceiling)
- App limits: 5 req/min per IP (X-Forwarded-For), 200 req/day, question <= 500 chars, top_n <= 10

## Cost safety
- Gemini key lives in a project WITHOUT billing (free tier only, 429 instead of charges)
- Budget "jarvis-rag-cap": Cloud Run spend cap 8,000 KRW/month (pauses service)
- Budget "jarvis-rag-alerts": all services, email at 50/90/100% of 10,000 KRW

## Incidents
- 2026-10-17 Embedding 400 "unexpected model name format":
  PowerShell turned the unquoted comma in --set-env-vars into a space, so two variables were stored as one.
  Fix: quote the value -> --update-env-vars "A=1,B=2". Service stayed up thanks to BM25 fallback.
- 2026-10-17 Gemini embedding free daily quota hit at 900/2,000 papers -> resume after quota reset.

## Logs
gcloud run services logs read jarvis-rag --region asia-northeast3 --limit 20