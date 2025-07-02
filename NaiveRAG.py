# pip install openai==1.93.0 faiss-cpu python-dotenv tiktoken
import os
import json
from dotenv import load_dotenv
import faiss
import numpy as np
import tiktoken
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from prompt.answer import ANSWER_PROMPT  # 여기에 your prompt 템플릿이 문자열로 정의되어 있어야 합니다

# === Configuration ===
CONTEXT_PATH    = 'hotpotQA/contexts.txt'
QUESTIONS_PATH  = 'hotpotQA/questions.json'
OUTPUT_PATH     = 'hotpotQA/naive_answers.json'
INDEX_PATH      = 'hotpotQA/faiss_index.bin'
CHUNKS_PATH     = 'hotpotQA/chunks.json'

# 1. Load API key and init client
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# 2. Parameters
CHUNK_SIZE      = 1200
CHUNK_OVERLAP   = 100
EMBEDDING_MODEL = 'text-embedding-3-small'
QA_MODEL        = 'gpt-4o-mini'
MAX_WORKERS     = 5

# 3. Load or build FAISS index + chunks
if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    # Load index and chunks
    print(f"Loading FAISS index from {INDEX_PATH} and chunks from {CHUNKS_PATH}...")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
else:
    # Read and chunk contexts
    with open(CONTEXT_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    encoder = tiktoken.get_encoding('cl100k_base')
    tokens = encoder.encode(text)
    chunks = [
        encoder.decode(tokens[i:i + CHUNK_SIZE])
        for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP)
    ]
    # Embed chunks in parallel
    def embed_chunk(chunk: str) -> np.ndarray:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=chunk)
        return np.array(resp.data[0].embedding, dtype='float32')

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        embeddings = list(tqdm(exe.map(embed_chunk, chunks), total=len(chunks), desc="Embedding chunks"))

    emb_matrix = np.vstack(embeddings)
    index = faiss.IndexFlatL2(emb_matrix.shape[1])
    index.add(emb_matrix)

    # Save index and chunks
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"Built and saved FAISS index to {INDEX_PATH} and chunks to {CHUNKS_PATH}")

# 4. Load questions and RAG
with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f:
    questions = json.load(f)
results = []

# Function to process one question
def process_question(item) -> dict:
    question = item.get('query') if isinstance(item, dict) else item
    # Embed question
    q_emb = client.embeddings.create(model=EMBEDDING_MODEL, input=question)
    q_vec = np.array(q_emb.data[0].embedding, dtype='float32').reshape(1, -1)
    # Retrieve top-3 chunks
    D, I = index.search(q_vec, 7)
    retrieved = [chunks[i] for i in I[0]]
    # Build prompt and generate answer
    context = "\n\n".join(retrieved)
    prompt = ANSWER_PROMPT.replace("{question}", question).replace("{context}", context)
    chat = client.chat.completions.create(
        model=QA_MODEL,
        messages=[{'role':'system','content':'You are a helpful assistant.'},
                  {'role':'user','content':prompt}]
    )
    return {"query": question, "answer": chat.choices[0].message.content.strip()}

# 5. Process questions in parallel
with ThreadPoolExecutor(max_workers=1) as exe:
    for res in tqdm(exe.map(process_question, questions), total=len(questions), desc="Processing questions"):
        results.append(res)

# 6. Save answers
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Completed RAG. Answers saved to {OUTPUT_PATH}")
