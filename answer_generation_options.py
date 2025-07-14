import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from graph_based_rag_options import GraphRAG  # the updated class above

# -------------------------------------------------------------------- paths
INPUT_PATH  = Path("InfiniteChoice/qa.json")  # must contain "query" and "options" keys
OUTPUT_PATH = Path("InfiniteChoice/result/kgrag_v1.json")
TEMP_PATH   = OUTPUT_PATH.with_stem(OUTPUT_PATH.stem + "_temp")

rag = GraphRAG()

# ---------------------------------------------------------------- load input
INPUT_PATH  = Path("InfiniteChoice/qa.json")        # must contain "query" key only (no options)
OPTIONS_PATH = Path("InfiniteChoice/options.json")        # external list-of-lists file provided by user

with INPUT_PATH.open("r", encoding="utf-8") as f:
    questions = json.load(f)            # List[Dict], each with at least "query"

with OPTIONS_PATH.open("r", encoding="utf-8") as f:
    all_options = json.load(f)          # List[List[str]] – exactly aligned with questions

if len(questions) != len(all_options):
    raise ValueError(
        f"Mismatched lengths: {len(questions)} questions vs {len(all_options)} option sets")

# Keep output order stable
output_data = [None] * len(questions)

# ---------------------------------------------------------------- worker fn

def _process(idx_and_item):
    idx, item = idx_and_item
    question = item.get("query", "")
    options = all_options[idx]  # retrieve the matching options list

    try:
        answer = rag.answer(question, options)
    except Exception as exc:
        answer = f"[Error] {exc}"

    return idx, {"query": question, "answer": answer}

# ---------------------------------------------------------------- run jobs
SAVE_EVERY = 100
completed   = 0

with ThreadPoolExecutor(max_workers=20) as ex:
    futures = [ex.submit(_process, (i, q)) for i, q in enumerate(questions)]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating answers"):
        idx, res = fut.result()
        output_data[idx] = res
        completed += 1

        if completed % SAVE_EVERY == 0:
            TEMP_PATH.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[Temp Save] {completed} items saved → {TEMP_PATH}")

# ---------------------------------------------------------------- save final
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved final output to {OUTPUT_PATH}")
