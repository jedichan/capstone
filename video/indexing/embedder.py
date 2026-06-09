import sys
from pathlib import Path
from typing import List, Union


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.tools.llm import get_embedding_runtime_profile, get_qwen_embedding

if __name__ == "__main__":
    test_text = "A silver-gray sedan parked stationary in a parking slot for a long time."
    print(f"Test embedding for: {test_text}")
    try:
        vec = get_qwen_embedding(test_text)
        print(f"OK: dim={len(vec)}")
        print(f"Runtime profile: {get_embedding_runtime_profile()}")
        print(f"First 5 values: {vec[:5]}")
    except Exception as e:
        print(f"Failed: {e}")
