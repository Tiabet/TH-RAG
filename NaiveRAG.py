import os
import json
from dotenv import load_dotenv
import faiss
import numpy as np
import tiktoken
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from prompt.answer import ANSWER_PROMPT

# === Configuration ===
CONTEXT_PATH    = 'hotpotQA/contexts.txt'
QUESTIONS_PATH  = 'hotpotQA/qa.json'
OUTPUT_PATH     = 'Result/NaiveRAG/hotpot_result.json'
INDEX_PATH      = 'Result/NaiveRAG/hotpot_chunks_faiss.bin'
CHUNKS_PATH     = 'Result/NaiveRAG/hotpot_chunks.json'
USED_CHUNKS_LOG = 'Result/NaiveRAG/hotpot_used_chunks.jsonl'
top_k = 7

# Load environment and OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = 'text-embedding-3-small'
QA_MODEL = 'gpt-4o-mini'
MAX_WORKERS = 30

# === Load or create FAISS index and chunk UUID map ===
if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    print(f"📥 Loading FAISS index from {INDEX_PATH} and chunks from {CHUNKS_PATH}...")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunk_kv = json.load(f)
    chunk_uuid_list = list(chunk_kv.keys())
    chunks = list(chunk_kv.values())
else:
    print("🔧 Building FAISS index and chunks...")
    with open(CONTEXT_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    encoder = tiktoken.get_encoding('cl100k_base')
    tokens = encoder.encode(text)
    chunks = [
        encoder.decode(tokens[i:i + CHUNK_SIZE])
        for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP)
    ]
    chunk_uuid_list = [f"chunk-{i:06d}" for i in range(len(chunks))]
    chunk_kv = dict(zip(chunk_uuid_list, chunks))

    def embed_chunk(chunk: str) -> np.ndarray:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=chunk)
        emb = np.array(resp.data[0].embedding, dtype='float32')
        return emb / np.linalg.norm(emb)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        embeddings = list(tqdm(exe.map(embed_chunk, chunks), total=len(chunks), desc="🔍 Embedding chunks"))

    emb_matrix = np.vstack(embeddings)
    index = faiss.IndexFlatIP(emb_matrix.shape[1])
    index.add(emb_matrix)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunk_kv, f, ensure_ascii=False)
    print(f"✅ Saved FAISS index to {INDEX_PATH} and chunks to {CHUNKS_PATH}")

# === Load questions ===
with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f:
    questions = json.load(f)
results = []
chunk_log_file = open(USED_CHUNKS_LOG, 'w', encoding='utf-8')

# === QA processing function ===
def process_question(item) -> dict:
    question = item.get('query') if isinstance(item, dict) else item
    q_emb = client.embeddings.create(model=EMBEDDING_MODEL, input=question)
    q_vec = np.array(q_emb.data[0].embedding, dtype='float32')
    q_vec /= np.linalg.norm(q_vec)
    q_vec = q_vec.reshape(1, -1)

    D, I = index.search(q_vec, top_k)
    retrieved_chunks = [chunks[i] for i in I[0]]
    used_chunk_ids = [chunk_uuid_list[i] for i in I[0]]

    for chunk_id in used_chunk_ids:
        chunk_log_file.write(json.dumps({
            "query": question,
            "chunk_id": chunk_id
        }, ensure_ascii=False) + "\n")

    context = "\n\n".join(retrieved_chunks)
    prompt = ANSWER_PROMPT.replace("{question}", question).replace("{context}", context)
    chat = client.chat.completions.create(
        model=QA_MODEL,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ]
    )

    return {
        "query": question,
        "result": chat.choices[0].message.content.strip(),
        "used_chunks": used_chunk_ids
    }

# === Execute RAG ===
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
    for res in tqdm(exe.map(process_question, questions), total=len(questions), desc="🤖 Processing questions"):
        results.append(res)

chunk_log_file.close()

# === Save results ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"🎉 Completed RAG. Answers saved to {OUTPUT_PATH}")
print(f"🧾 Used chunk log saved to {USED_CHUNKS_LOG}")
