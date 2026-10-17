# ===== Check the Gemini API key: list models, test one embedding and one answer =====
# Usage: python scripts/gemini_check.py


import os

from dotenv import load_dotenv
from google import genai


# Read GEMINI_API_KEY from the .env file into environment variables
load_dotenv()
key = os.getenv("GEMINI_API_KEY")
if not key:
    raise SystemExit("GEMINI_API_KEY not found. Check the .env file in the project root.")

client = genai.Client(api_key=key)

# ===== 1) Models available to this key =====
print("== Available models (embedding / flash) ==")
for m in client.models.list():
    if "embed" in m.name or "flash" in m.name:
        print("  ", m.name)

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-flash-latest")

# ===== 2) One embedding (public text only) =====
res = client.models.embed_content(model=EMBED_MODEL, contents=["chest X-ray pneumonia detection"])
vector = res.embeddings[0].values
print(f"\nEmbedding OK: {EMBED_MODEL}  dimensions={len(vector)}")

# ===== 3) One short answer =====
out = client.models.generate_content(model=LLM_MODEL, contents="RAG를 한 문장으로 설명해줘.")
print(f"LLM OK: {LLM_MODEL}\n{out.text}")


# → .env에서 키를 읽어 Gemini에 연결하고, 쓸 수 있는 모델 목록 → 임베딩 1회 → 짧은 답변 1회를 시험하겠다.
#   키 값 자체는 화면에 출력하지 않는다.