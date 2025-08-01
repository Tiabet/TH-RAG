import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import openai
import tiktoken
import argparse

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "prompt"))

# 설정 및 프롬프트 import
from config import get_config
from prompt.topic_choice import get_topic_choice_prompt

# ==== Configuration ====
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Note: API key will be checked when actually needed, not at import time

MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 3000
OVERLAP = 300  
MAX_WORKERS = 10

# ==== Functions ====
def chunk_text(text, max_tokens, overlap, model_name):
    """
    텍스트를 지정된 토큰 수에 맞게 청크로 나눕니다.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        if end >= len(tokens):
            break
            
        start = end - overlap
    
    return chunks

def call_model(client, model_name, chunk, index):
    """
    OpenAI API를 호출하여 주어진 청크에 대한 QA를 생성합니다.
    """
    try:
        prompt = get_topic_choice_prompt()
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": chunk}
            ],
            temperature=0.5,
            max_tokens=2000
        )
        data = json.loads(response.choices[0].message.content.strip())
        if isinstance(data, list):
            for item in data:
                item["chunk_id"] = index
        elif isinstance(data, dict):
            data["chunk_id"] = index
        return data
    except Exception as e:
        return {"error": str(e), "chunk_index": index}

# ==== Main Function ====
def main(dataset_name: str = None, input_path_param: str = None, output_path_param: str = None):
    """
    Main function to run graph construction.
    
    Args:
        dataset_name: 데이터셋 이름
        input_path_param: 입력 파일 경로 (선택사항)
        output_path_param: 출력 파일 경로 (선택사항)
    """
    # Check API key when actually running
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    # 설정 객체 생성
    if not dataset_name:
        raise ValueError("Dataset name is required")
    
    config = get_config(dataset_name)
    
    # 경로 설정
    current_input_path = input_path_param if input_path_param else str(config.get_input_file())
    current_output_path = output_path_param if output_path_param else str(config.get_qa_file())
    
    print(f"📂 Processing dataset: {dataset_name}")
    print(f"📂 Input: {current_input_path}")
    print(f"💾 Output: {current_output_path}")
    
    # ==== Load Text & Chunk ====
    full_text = Path(current_input_path).read_text(encoding="utf-8")
    chunks = chunk_text(full_text, MAX_TOKENS, OVERLAP, MODEL_NAME)
    print(f"📄 Created {len(chunks)} chunks")

    # ==== Load Existing ====
    output_path_obj = Path(current_output_path)
    if output_path_obj.exists():
        with open(output_path_obj, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
                print(f"🔄 Loaded {len(results)} existing results.")
            except json.JSONDecodeError:
                print("⚠️ JSON decode error. Starting fresh.")
                results = [None] * len(chunks)
    else:
        results = [None] * len(chunks)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    pending_indices = [i for i, r in enumerate(results) if r is None or "error" in r]
    print(f"🕐 Remaining chunks to process: {len(pending_indices)}")

    if not pending_indices:
        print("✅ All chunks already processed!")
        return

    # ==== Run ====
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(call_model, client, MODEL_NAME, chunks[idx], idx): idx
            for idx in pending_indices
        }

        for future in tqdm(as_completed(futures), total=len(pending_indices), desc="Processing"):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result
                # 매 10개마다 저장
                if idx % 10 == 0:
                    with open(output_path_obj, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"❌ Error for chunk {idx}: {e}")
                results[idx] = {"error": str(e), "chunk_index": idx}

    # ==== Final Save ====
    with open(output_path_obj, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Completed! Results saved to {current_output_path}")
    
    # 파이프라인 상태 업데이트
    state = config.load_pipeline_state() or {}
    state[dataset_name] = state.get(dataset_name, {})
    state[dataset_name]['graph_construction'] = {
        'completed': True,
        'input_file': current_input_path,
        'output_file': current_output_path,
        'qa_file': str(config.get_qa_file())
    }
    config.save_pipeline_state(state)
    
    return str(config.get_qa_file())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph construction for KGRAG")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--input", help="Input contexts.txt file path")
    parser.add_argument("--output", help="Output QA JSON file path")
    
    args = parser.parse_args()
    main(args.dataset, args.input, args.output)
