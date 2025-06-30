def remove_duplicates_and_count(file_path):
    # 파일에서 줄별로 문장 읽기 (양 끝 공백 제거)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    original_count = len(lines)
    
    # 중복 제거
    unique_lines = list(set(lines))
    unique_count = len(unique_lines)
    
    # 결과 출력
    print(f"중복 제거 전 문장 개수: {original_count}")
    print(f"중복 제거 후 문장 개수: {unique_count}")
    print(f"중복된 문장 개수: {original_count - unique_count}")
    
    # 원하면 중복 제거된 문장을 새 파일로 저장 가능
    with open('hotpotQA/fullwiki_contexts_deduplicated.txt', 'w', encoding='utf-8') as out:
        out.write('\n'.join(unique_lines))

# 사용 예시
remove_duplicates_and_count('hotpotQA/fullwiki_contexts.txt')