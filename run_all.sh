# #!/bin/bash

# # set -e  # 오류 발생 시 즉시 종료

# echo "🔄 [1/4] Converting JSON to GEXF..."
# for path in UltraDomain/CS UltraDomain/Mix 
# do
#   echo "➡️  Converting: $path/graph_v1.json"
#   python json_to_gexf.py "$path/graph_v1.json"
#   echo "✅ Done: $path/graph_v1.json → GEXF"
# done

# echo ""
# echo "🔄 [2/4] Normalizing topics/subtopics..."
# for path in UltraDomain/CS UltraDomain/Mix 
# do
#   echo "➡️  Normalizing: $path/graph_v1.gexf"
#   python normalize_topics.py -i "$path/graph_v1.gexf" -o "$path/graph_v1_processed.gexf"
#   echo "✅ Done: $path/graph_v1_processed.gexf"
# done

# echo ""
# echo "🔄 [3/3] Building FAISS edge embeddings..."
# for path in UltraDomain/Agriculture UltraDomain/CS UltraDomain/Mix UltraDomain/Legal MultihopRAG
# do
#   echo "➡️  Embedding: $path/graph_v1_processed.gexf"
#   python edge_embedding.py \
#     -g "$path/graph_v1_processed.gexf" \
#     -i "$path/edge_index_v1.faiss" \
#     -p "$path/edge_payloads_v1.npy"
#   echo "✅ Done: $path FAISS + Payload"
# done

# echo ""
# echo "🎉 전체 파이프라인 완료!"
  python edge_embedding.py \
    -g "hotpotQA/graph_v1_processed.gexf" \
    -i "hotpotQA/edge_index_v1.faiss" \
    -p "hotpotQA/edge_payloads_v1.npy"
  echo "✅ Done: hotpotQA FAISS + Payload"