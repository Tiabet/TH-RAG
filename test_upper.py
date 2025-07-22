#!/usr/bin/env python
import os, json, re, string, time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from openai import RateLimitError
from tqdm import tqdm
from prompt.answer_short import ANSWER_PROMPT
from dotenv import load_dotenv
load_dotenv()
import tiktoken
ENC = tiktoken.encoding_for_model("gpt-4o-mini")   # 모델별 토크나이저

def num_tokens(text: str) -> int:
    return len(ENC.encode(text))

# ---------- 설정 ----------
CHUNKS_PATH  = Path("hotpotQA/chunks.txt")
HOTPOT_PATH  = Path("hotpotQA/sampled_200.jsonl")
N_THREADS    = 50                # 필요 시 조절
MODEL        = "gpt-4o-mini"
MAX_TOK_CTX  = 8192
SLEEP_ERR    = 5

# ---------- 유틸 ----------
def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())

def compute_em(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))

# ---------- 1) 데이터 로드 ----------
chunks = [line.rstrip("\n") for line in CHUNKS_PATH.open(encoding="utf-8")]
questions = [json.loads(l) for l in HOTPOT_PATH.open(encoding="utf-8")]
print(f"💾 {len(chunks):,} chunks | {len(questions):,} questions")

# ---------- 2) 역색인 ----------
word2chunks = defaultdict(set)
for idx, chunk in enumerate(chunks):
    for w in normalize(chunk).split():
        word2chunks[w].add(idx)

def map_chunks(item):
    ctx = {t: s for t, s in item["context"]}
    hit = set()
    for title, sent_no in item["supporting_facts"]:
        try:
            sent   = ctx[title][sent_no]
            words  = normalize(sent).split()
            rare_w = min(words, key=lambda w: len(word2chunks[w]))
            for idx in word2chunks[rare_w]:
                if " ".join(words) in normalize(chunks[idx]):
                    hit.add(idx)
        except (KeyError, IndexError):
            pass
    return sorted(hit)

# ---------- 3) 프롬프트 ----------
def build_prompt(q, idxs):
    selected, tot = [], 0
    for i in idxs:
        chunk = chunks[i]
        est = num_tokens(chunk)
        if tot + est > MAX_TOK_CTX * 3 // 4:
            break
        selected.append(f"[{i}] {chunk}")
        tot += est
    prompt = ANSWER_PROMPT.replace("{{question}}", q)\
                          .replace("{{context}}", "\n".join(selected))
    print(f"⋯ prompt tokens = {num_tokens(prompt)}")
    return prompt


# ---------- 4) OpenAI ----------
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(prompt):
    while True:
        try:
            r = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0)
            return r.choices[0].message.content.strip()
        except RateLimitError:
            print("⏳ rate‑limit… sleep", SLEEP_ERR)
            time.sleep(SLEEP_ERR)
        except Exception as e:
            print("⚠️", e); return ""

# ---------- 5) worker ----------
def worker(item):
    idxs = map_chunks(item)
    if not idxs:
        return {"query": item["question"], "result": "", "em": 0}

    pred = call_llm(build_prompt(item["question"], idxs))
    em   = compute_em(pred, item["answer"])
    return {"query": item["question"], "result": pred, "em": em}

# ---------- 6) 실행 ----------
raw_results, em_sum = [], 0
with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
    futures = {ex.submit(worker, q): q for q in questions}
    for f in tqdm(as_completed(futures), total=len(futures)):
        res = f.result()
        raw_results.append(res)
        em_sum += res["em"]

# ---------- 7) 리포트 & 저장 ----------
print("\n🔎 Quick audit (20개)")
for it in questions[:20]:
    print(f"SF {len(it['supporting_facts'])} → chunks {len(map_chunks(it))}")

acc = em_sum / len(questions)
print(f"\n✅ EM accuracy: {acc:.3%}")

# query/result만 추려서 저장
out_data = [{"query": r["query"], "result": r["result"]} for r in raw_results]
Path("llm_preds.json").write_text(json.dumps(out_data,
                                             ensure_ascii=False,
                                             indent=2),
                                  encoding="utf-8")
print("📝 saved → llm_preds.json")
