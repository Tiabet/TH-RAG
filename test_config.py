#!/usr/bin/env python3
"""
KGRAG 설정 테스트 스크립트
.env 파일의 설정들이 올바르게 로드되는지 확인합니다.
"""

import os
from pathlib import Path

def test_config():
    """설정 테스트"""
    print("🔧 KGRAG Configuration Test")
    print("=" * 50)
    
    # .env 파일 확인
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print("⚠️  .env 파일이 없습니다. .env.example을 복사해서 설정하세요:")
            print(f"   cp {env_example} {env_file}")
        else:
            print("❌ .env.example 파일도 없습니다!")
        return
    
    # Config 로드 테스트
    try:
        from config import get_config
        config = get_config()
        print("✅ Config 로드 성공!")
        
        # API 키 확인
        print(f"\n📍 API 설정:")
        if config.openai_api_key:
            print(f"   OpenAI API Key: {'*' * 10}{config.openai_api_key[-4:] if len(config.openai_api_key) > 4 else '****'}")
        else:
            print("   ⚠️  OpenAI API Key가 설정되지 않았습니다!")
            
        # 모델 설정
        print(f"\n🤖 Model 설정:")
        print(f"   기본 모델: {config.default_model}")
        print(f"   임베딩 모델: {config.embed_model}")
        print(f"   채팅 모델: {config.chat_model}")
        print(f"   평가 모델: {config.eval_model}")
        
        # 하이퍼파라미터
        print(f"\n⚙️  하이퍼파라미터:")
        print(f"   Temperature: {config.temperature}")
        print(f"   Max Tokens: {config.max_tokens}")
        print(f"   Overlap: {config.overlap}")
        print(f"   Top-K1: {config.top_k1}")
        print(f"   Top-K2: {config.top_k2}")
        
        # 토픽/서브토픽 설정
        print(f"\n📋 토픽 설정:")
        print(f"   Topic 선택 범위: {config.topic_choice_min}-{config.topic_choice_max}")
        print(f"   Subtopic 선택 범위: {config.subtopic_choice_min}-{config.subtopic_choice_max}")
        
        # 시스템 설정
        print(f"\n🔧 시스템 설정:")
        print(f"   Max Workers: {config.max_workers}")
        print(f"   Log Level: {config.log_level}")
        print(f"   Batch Size: {config.batch_size}")
        
        print(f"\n✅ 모든 설정이 정상적으로 로드되었습니다!")
        
    except Exception as e:
        print(f"❌ Config 로드 실패: {e}")
        return
    
    # 환경 변수 직접 확인
    print(f"\n🌍 환경 변수 확인:")
    env_vars = [
        "OPENAI_API_KEY", "DEFAULT_MODEL", "EMBED_MODEL", "CHAT_MODEL",
        "TEMPERATURE", "MAX_TOKENS", "TOP_K1", "TOP_K2",
        "TOPIC_CHOICE_MIN", "TOPIC_CHOICE_MAX"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if "API_KEY" in var:
                display_value = f"{'*' * 10}{value[-4:] if len(value) > 4 else '****'}"
            else:
                display_value = value
            print(f"   {var}: {display_value}")
        else:
            print(f"   {var}: ❌ 설정되지 않음")

def create_sample_env():
    """샘플 .env 파일 생성"""
    print("\n📝 샘플 .env 파일 생성 중...")
    
    env_content = """# KGRAG Configuration
# OpenAI API Key (필수)
OPENAI_API_KEY=your_openai_api_key_here

# Model Settings
DEFAULT_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
EVAL_MODEL=gpt-4o-mini

# Generation Parameters
TEMPERATURE=0.5
MAX_TOKENS_RESPONSE=2000
ANSWER_TEMPERATURE=0.3
ANSWER_MAX_TOKENS=1000
EVAL_TEMPERATURE=0.1

# Text Processing
MAX_TOKENS=3000
OVERLAP=300
MAX_WORKERS=10

# Topic Selection
TOPIC_CHOICE_MIN=5
TOPIC_CHOICE_MAX=10
SUBTOPIC_CHOICE_MIN=10
SUBTOPIC_CHOICE_MAX=25

# RAG Parameters
TOP_K1=50
TOP_K2=10
EMBEDDING_TOP_K=5
OVERRETRIEVE_FACTOR=5

# System Settings
LOG_LEVEL=INFO
BATCH_SIZE=32
TIMEOUT_SECONDS=30
"""
    
    with open(".env", "w", encoding="utf-8") as f:
        f.write(env_content)
    
    print("✅ .env 파일이 생성되었습니다!")
    print("⚠️ OPENAI_API_KEY를 실제 API 키로 교체해주세요!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="KGRAG 설정 테스트")
    parser.add_argument("--create-env", action="store_true", help="샘플 .env 파일 생성")
    
    args = parser.parse_args()
    
    if args.create_env:
        create_sample_env()
    else:
        test_config()
