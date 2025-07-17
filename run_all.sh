#!/bin/bash

# 처리할 도메인 목록
DOMAINS=("Mix" "CS" "Legal")

for DOMAIN in "${DOMAINS[@]}"; do
  INPUT_FILE="UltraDomain/$DOMAIN/${DOMAIN,,}_unique_contexts.txt"  # 소문자로 변환
  OUTPUT_FILE="UltraDomain/$DOMAIN/graph_v1.json"

  echo "🔹 Processing $DOMAIN"

  if [ -f "$INPUT_FILE" ]; then
    python graph_construction.py --input "$INPUT_FILE" --output "$OUTPUT_FILE"
    echo "✅ Finished $DOMAIN"
  else
    echo "⚠️  Skipping $DOMAIN: $INPUT_FILE not found."
  fi
done
