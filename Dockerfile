# ===== Jarvis RAG API image: code + public model only, NO private data =====

# Base: small official Python 3.12 Linux image (same Python version as the laptop)
FROM python:3.12-slim

# Don't create .pyc files; print logs immediately instead of buffering
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# 1) CPU-only PyTorch. The default Linux torch also bundles GPU (CUDA) libraries worth several GB
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

# 2) Other libraries. requirements.txt is copied before the code, so this slow step is reused (cached) when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Download the public embedding model at build time, so the container does not download it on every start
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# 4) App code only (scripts, data and .venv are excluded by .dockerignore)
COPY app ./app

# Private data is mounted at /data when the container runs; Ollama stays on the laptop
ENV JARVIS_DATA_DIR=/data \
    OLLAMA_URL=http://host.docker.internal:11434/api/chat

EXPOSE 8000

# 0.0.0.0 here means "listen inside the container's own network".
# Exposure to the outside is limited later by "docker run -p 127.0.0.1:8000:8000"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]