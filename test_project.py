#!/usr/bin/env python3
"""
KGRAG 프로젝트 구조 및 기능 테스트 스크립트
"""

import sys
import os
from pathlib import Path

def test_project_structure():
    """프로젝트 구조 테스트"""
    print("🏗️ 프로젝트 구조 테스트")
    print("=" * 50)
    
    expected_dirs = ["index", "generate", "evaluate", "prompt"]
    project_root = Path(".")
    
    for dir_name in expected_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ 폴더 존재")
            
            # 각 폴더의 주요 파일들 확인
            if dir_name == "index":
                files = ["graph_construction.py", "build_index.bat", "build_index.sh"]
            elif dir_name == "generate":
                files = ["Retriever.py", "generate_answers.bat", "generate_answers.sh"]
            elif dir_name == "evaluate":
                files = ["judge_F1.py", "evaluate_answers.bat", "evaluate_answers.sh"]
            elif dir_name == "prompt":
                files = ["topic_choice.py"]
            
            for file_name in files:
                file_path = dir_path / file_name
                if file_path.exists():
                    print(f"  ✅ {file_name}")
                else:
                    print(f"  ❌ {file_name} 누락")
        else:
            print(f"❌ {dir_name}/ 폴더 누락")
    
    print()

def test_imports():
    """Import 테스트"""
    print("📦 모듈 Import 테스트")
    print("=" * 50)
    
    # 프로젝트 루트를 sys.path에 추가
    project_root = Path(".").resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "index"))
    sys.path.insert(0, str(project_root / "generate"))
    sys.path.insert(0, str(project_root / "evaluate"))
    sys.path.insert(0, str(project_root / "prompt"))
    
    test_cases = [
        ("index.graph_construction", "main"),
        ("generate.Retriever", "Retriever"),
        ("evaluate.judge_F1", "main"),
        ("prompt.topic_choice", "get_topic_choice_prompt"),
    ]
    
    for module_name, function_name in test_cases:
        try:
            module = __import__(module_name, fromlist=[function_name])
            getattr(module, function_name)
            print(f"✅ {module_name}.{function_name}")
        except ImportError as e:
            print(f"❌ {module_name}.{function_name} - Import Error: {e}")
        except AttributeError as e:
            print(f"❌ {module_name}.{function_name} - Attribute Error: {e}")
        except Exception as e:
            print(f"⚠️ {module_name}.{function_name} - Other Error: {e}")
    
    print()

def test_shell_scripts():
    """Shell 스크립트 존재 확인"""
    print("🚀 실행 스크립트 테스트")
    print("=" * 50)
    
    scripts = [
        "run_all.sh",
        "run_all.bat", 
        "index/build_index.sh",
        "index/build_index.bat",
        "generate/generate_answers.sh", 
        "generate/generate_answers.bat",
        "evaluate/evaluate_answers.sh",
        "evaluate/evaluate_answers.bat"
    ]
    
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            print(f"✅ {script}")
        else:
            print(f"❌ {script} 누락")
    
    print()

def test_dependencies():
    """의존성 패키지 테스트"""
    print("📋 의존성 패키지 테스트")
    print("=" * 50)
    
    packages = [
        "openai",
        "faiss",  # faiss-cpu가 설치되면 faiss로 import됨
        "networkx",
        "numpy",
        "pandas",
        "tiktoken",
        "tqdm"
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} 누락")
    
    print()

def main():
    """메인 테스트 실행"""
    print("🧪 KGRAG 프로젝트 종합 테스트")
    print("=" * 70)
    print()
    
    test_project_structure()
    test_imports() 
    test_shell_scripts()
    test_dependencies()
    
    print("🎯 테스트 완료!")
    print("=" * 70)
    print()
    print("✨ 다음 단계:")
    print("1. 환경 변수 OPENAI_API_KEY 설정")
    print("2. 데이터셋 폴더 준비 (예: hotpotQA/, UltraDomain/)")
    print("3. run_all.bat 또는 run_all.sh 실행")

if __name__ == "__main__":
    main()
