import json

input_file = "InfiniteChoice/infinitechoice.jsonl"
output_file = "InfiniteChoice/unique_contexts.txt"

unique_contexts = set()

with open(input_file, "r", encoding="utf-8") as f_in:
    for line in f_in:
        data = json.loads(line)
        context = data.get("context", "").strip()
        if context:
            unique_contexts.add(context)

with open(output_file, "w", encoding="utf-8") as f_out:
    for context in unique_contexts:
        f_out.write(context + "\n")
