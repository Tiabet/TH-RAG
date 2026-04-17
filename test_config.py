"""Utility script for validating TH-RAG configuration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def test_config() -> None:
    print("TH-RAG configuration check")
    print("=" * 60)

    env_path = Path(".env")
    env_example_path = Path(".env.example")

    if not env_path.exists():
        if env_example_path.exists():
            print("Missing .env file. Copy .env.example and fill in your API key.")
            print(f"Example: copy {env_example_path} {env_path}")
        else:
            print("Missing both .env and .env.example.")
        return

    try:
        from config import get_config

        config = get_config()
    except Exception as exc:
        print(f"Failed to load configuration: {exc}")
        return

    print("API settings")
    print("-" * 60)
    if config.openai_api_key:
        masked_key = f"{'*' * 8}{config.openai_api_key[-4:]}" if len(config.openai_api_key) >= 4 else "configured"
        print(f"OPENAI_API_KEY: {masked_key}")
    else:
        print("OPENAI_API_KEY: missing")

    print("\nModel settings")
    print("-" * 60)
    print(f"DEFAULT_MODEL: {config.default_model}")
    print(f"EMBED_MODEL: {config.embed_model}")
    print(f"CHAT_MODEL: {config.chat_model}")
    print(f"EVAL_MODEL: {config.eval_model}")

    print("\nRetrieval and generation settings")
    print("-" * 60)
    print(f"TOP_K1 / TOP_K2: {config.top_k1} / {config.top_k2}")
    print(f"TOP_K1_LONG / TOP_K2_LONG: {config.top_k1_long} / {config.top_k2_long}")
    print(f"MAX_TOKENS / OVERLAP: {config.max_tokens} / {config.overlap}")
    print(f"MAX_WORKERS: {config.max_workers}")

    print("\nEnvironment variables")
    print("-" * 60)
    for name in [
        "OPENAI_API_KEY",
        "DEFAULT_MODEL",
        "EMBED_MODEL",
        "CHAT_MODEL",
        "EVAL_MODEL",
        "TOP_K1",
        "TOP_K2",
        "MAX_TOKENS",
        "OVERLAP",
    ]:
        value = os.getenv(name)
        if not value:
            print(f"{name}: missing")
        elif "API_KEY" in name:
            print(f"{name}: {'*' * 8}{value[-4:]}")
        else:
            print(f"{name}: {value}")



def create_sample_env() -> None:
    template = Path(".env.example")
    if not template.exists():
        raise FileNotFoundError(".env.example is missing.")
    Path(".env").write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    print("Created .env from .env.example. Update OPENAI_API_KEY before running the pipeline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect TH-RAG configuration values.")
    parser.add_argument("--create-env", action="store_true", help="Create .env from .env.example")
    args = parser.parse_args()

    if args.create_env:
        create_sample_env()
    else:
        test_config()
