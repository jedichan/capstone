"""Quick API quota/connectivity test — makes one minimal call to qwen3.6-plus."""
import os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    print("[FAIL] DASHSCOPE_API_KEY not found in environment")
    sys.exit(1)

from openai import OpenAI
client = OpenAI(
    api_key=api_key,
    base_url=os.environ.get("DASHSCOPE_URL",
                             "https://dashscope.aliyuncs.com/compatible-mode/v1")
)

model = "qwen3.6-plus"
print(f"Testing model: {model}")
print(f"API key: {api_key[:8]}...{api_key[-4:]}")

try:
    resp = client.chat.completions.create(
        model=model,
        max_tokens=32,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}]
    )
    print(f"Response   : {resp.choices[0].message.content}")
    print(f"Tokens used: {resp.usage.total_tokens}")
    print(f"\n[PASS] API is working, quota available.")
except Exception as e:
    print(f"\n[FAIL] {type(e).__name__}: {e}")
    sys.exit(1)
