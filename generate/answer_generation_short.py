import json, sys, os
from pathlib import Path
import argparse
from graph_based_rag_short import GraphRAG
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken

# Set project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Import configuration
from config import get_config

enc = tiktoken.encoding_for_model("gpt-4o")

# Default settings
MAX_WORKERS = 30  # Number of parallel processing threads
TOP_K1 = 30
TOP_K2 = 5

def main(dataset_name: str, input_path_param: str = None, output_path_param: str = None):
    """
    Main function for answer generation (short)
    
    Args:
        dataset_name: Dataset name
        input_path_param: Input file path (optional)
        output_path_param: Output file path (optional)
    """
    config = get_config(dataset_name)
    
    # Path configuration
    input_path = input_path_param if input_path_param else str(config.get_qa_file())
    output_path = output_path_param if output_path_param else str(config.get_answer_file(answer_type="short"))
    chunk_log_path = str(config.get_chunk_log_file(answer_type="short"))
    temp_output_path = output_path.replace(".json", "_temp.json")
    
    print(f"📂 Processing dataset: {dataset_name}")
    print(f"📂 Input: {input_path}")
    print(f"💾 Output: {output_path}")
    
    # Create result directories
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(chunk_log_path), exist_ok=True)

    # GraphRAG instance
    rag = GraphRAG(dataset_name=dataset_name)

    chunk_log_file = open(chunk_log_path, "w", encoding="utf-8")

    # Load input
    with open(input_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # Result storage list (preserve index order)
    output_data = [None] * len(questions)

    # 작업 함수
    def process(index_query):
        idx, item = index_query
        query = item.get("query", "")
        try:
            answer, spent, context_token = rag.answer(query=query, top_k1=TOP_K1, top_k2=TOP_K2)
            chunk_ids = getattr(rag, 'last_chunk_ids', [])  # GraphRAG에서 마지막 chunk ID 기록하도록 추가 필요
        except Exception as e:
            answer = f"[Error] {e}"
            spent = 0.0
            context_token = None
            chunk_ids = []

        # 기록
        for cid in chunk_ids:
            log_entry = {"query": query, "chunk_id": cid}
            chunk_log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        result = {
            "query": query,
            "result": answer,
            "meta": {
                "total_spent": spent,
                "context_token": context_token
            }
        }
        return idx, result

    # Sentence chunk IDs logging (추가적)
    sentence_chunk_ids = set(getattr(rag, "all_sentence_chunk_ids", []))
    for cid in sentence_chunk_ids:
        log_entry = {"query": "global", "sentence_chunk_id": cid}
        chunk_log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    # 병렬 처리
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process, (i, item)) for i, item in enumerate(questions)]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            idx, result = future.result()
            output_data[idx] = result  # 순서 유지
            
            # 중간 저장 (10개마다)
            if idx % 10 == 0:
                with open(temp_output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

    chunk_log_file.close()

    # 최종 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved final output to {output_path}")

    # 통계
    valid_items = [it for it in output_data if it and not it["result"].startswith("[Error]")]
    print(f"Total: {len(output_data)}, Valid: {len(valid_items)}")
    
    # 파이프라인 상태 업데이트
    state = config.load_pipeline_state() or {}
    state[dataset_name] = state.get(dataset_name, {})
    state[dataset_name]['answer_generation_short'] = {
        'completed': True,
        'input_file': input_path,
        'output_file': output_path,
        'chunk_log_file': chunk_log_path,
        'total_questions': len(output_data),
        'valid_answers': len(valid_items)
    }
    config.save_pipeline_state(state)
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer generation (short) for KGRAG")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--input", help="Input QA JSON file path")
    parser.add_argument("--output", help="Output answers JSON file path")
    
    args = parser.parse_args()
    main(args.dataset, args.input, args.output)

