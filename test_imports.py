import sys
import os
from pathlib import Path

# 현재 파일의 위치 확인
print(f"Current file: {__file__}")

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent
print(f"Project root: {PROJECT_ROOT}")

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(PROJECT_ROOT))

# 각 모듈 폴더를 sys.path에 추가
sys.path.insert(0, str(PROJECT_ROOT / "index"))
sys.path.insert(0, str(PROJECT_ROOT / "generate"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluate"))
sys.path.insert(0, str(PROJECT_ROOT / "prompt"))

print("Current sys.path:")
for i, path in enumerate(sys.path[:10]):  # 처음 10개만 출력
    print(f"  {i}: {path}")

# 각 모듈 import 테스트
print("\n=== Import Tests ===")

try:
    from index.graph_construction import main as graph_main
    print("✅ index.graph_construction import successful")
except ImportError as e:
    print(f"❌ index.graph_construction import failed: {e}")

try:
    from generate.Retriever import Retriever
    print("✅ generate.Retriever import successful")
except ImportError as e:
    print(f"❌ generate.Retriever import failed: {e}")

try:
    from evaluate.judge_F1 import main as judge_main
    print("✅ evaluate.judge_F1 import successful")
except ImportError as e:
    print(f"❌ evaluate.judge_F1 import failed: {e}")

try:
    from prompt.topic_choice import get_topic_choice_prompt
    print("✅ prompt.topic_choice import successful")
except ImportError as e:
    print(f"❌ prompt.topic_choice import failed: {e}")

print("\n=== Test completed ===")
