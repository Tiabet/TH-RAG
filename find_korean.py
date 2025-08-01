#!/usr/bin/env python3
import os
import re

def find_korean_in_files():
    korean_files = []
    korean_pattern = re.compile(r'[가-힣]')
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if korean_pattern.search(content):
                            korean_files.append(file_path)
                except Exception as e:
                    continue
    
    for file_path in sorted(korean_files):
        print(f'{file_path}')

if __name__ == "__main__":
    find_korean_in_files()
