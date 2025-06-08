import argparse
import json
from pathlib import Path
from typing import List

import openai
import tiktoken

from prompt import PROMPT


def chunk_text(text: str, max_tokens: int, overlap: int, model: str) -> List[str]:
    """Split text into token chunks with overlap."""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk = enc.decode(chunk_tokens)
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def call_model(client: openai.OpenAI, model: str, chunk: str) -> dict:
    """Call the OpenAI chat completion API with the prompt and chunk."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": chunk},
        ],
        temperature=0,
    )
    text = response.choices[0].message.content
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": text}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract knowledge graph using GPT-4o-mini.")
    parser.add_argument("inputs", nargs="+", help="Text files to process")
    parser.add_argument("-o", "--output", default="graph.json", help="Output JSON file")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--max_tokens", type=int, default=1200, help="Tokens per chunk")
    parser.add_argument("--overlap", type=int, default=100, help="Token overlap between chunks")
    args = parser.parse_args()

    texts = [Path(p).read_text() for p in args.inputs]
    full_text = "\n".join(texts)

    chunks = chunk_text(full_text, args.max_tokens, args.overlap, args.model)

    client = openai.OpenAI()

    results = [call_model(client, args.model, c) for c in chunks]

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
