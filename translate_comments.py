#!/usr/bin/env python3
import os
import re

def translate_korean_comments():
    """Translate Korean comments to English in all Python files"""
    
    # Dictionary for common Korean-to-English translations
    translations = {
        # Comments and documentation
        '프로젝트 루트': 'project root',
        '프로젝트 루트를 경로에 추가': 'Add project root to path',
        '프로젝트 루트 설정': 'Set project root',
        '프로젝트 루트로 작업 디렉터리 변경': 'Change working directory to project root',
        '설정': 'Configuration',
        '설정 import': 'Import configuration',
        '설정 로드': 'Load configuration',
        '질문 로딩': 'Load questions',
        '결과 리스트 초기화': 'Initialize result list',
        '처리 함수': 'Processing function',
        '결과 디렉터리 생성': 'Create result directories',
        '인스턴스 생성': 'Create instance',
        '인코더 초기화': 'Initialize encoder',
        '하드코딩/경로': 'Hardcoded/Paths',
        '청크 로그': 'chunk log',
        '문장 기반 청크 로그': 'Sentence-based chunk log',
        '메인 실행': 'Main execution',
        '동적 경로 설정': 'Dynamic path configuration',
        '동적으로 수정하여 실행': 'Dynamically modify and execute',
        '임시 스크립트 파일 생성 및 실행': 'Create and execute temporary script file',
        '기존 결과 로드': 'Load existing results',
        '있다면': 'if any',
        '총': 'Total',
        '개 청크로 분할': ' chunks split',
        '개 청크 처리 중': ' chunks processing',
        '트리플 추출 완료': 'Triple extraction completed',
        '변환': 'conversion',
        '생성': 'Create',
        '인덱스 생성': 'Create index',
        '입력 파일을 찾을 수 없습니다': 'Input file not found',
        '환경 변수': 'environment variable',
        '를 설정해야 합니다': ' must be set',
        '또는': 'or',
        '본문만 추출': 'extract body only',
        '평가를 수행하고': 'perform evaluation and',
        '반환': 'return',
        '한 쌍': 'one pair'
    }
    
    korean_pattern = re.compile(r'[가-힣]')
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py') and file != 'find_korean.py' and file != 'translate_comments.py':
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if korean_pattern.search(content):
                        print(f"Processing: {file_path}")
                        
                        # Apply basic translations
                        modified = False
                        for korean, english in translations.items():
                            if korean in content:
                                content = content.replace(korean, english)
                                modified = True
                        
                        # Additional pattern-based replacements
                        patterns = [
                            (r'"""([^"]*[가-힣][^"]*)"""', r'"""(Korean comment - needs manual translation)"""'),
                            (r'# ([^#\n]*[가-힣][^#\n]*)', r'# (Korean comment - needs manual translation)'),
                            (r'print\(f"([^"]*[가-힣][^"]*)"\)', r'print(f"(Korean message - needs manual translation)")'),
                        ]
                        
                        for pattern, replacement in patterns:
                            if re.search(pattern, content):
                                content = re.sub(pattern, replacement, content)
                                modified = True
                        
                        if modified:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(content)
                            print(f"  ✅ Updated: {file_path}")
                        
                except Exception as e:
                    print(f"  ❌ Error processing {file_path}: {e}")

if __name__ == "__main__":
    translate_korean_comments()
