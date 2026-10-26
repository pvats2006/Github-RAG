"""
Code Repository RAG — production entry point.

Run:
    python -m app.main

Workflow:
    1. User enters a GitHub repository URL.
    2. System validates, clones, chunks, embeds, and indexes the repository (once per session/cache).
    3. Embedding model, FAISS index, retriever, and Groq LLM client are initialized once.
    4. User asks multiple questions in an interactive loop.
    5. Every answer includes:
         - Groq answer with inline citations [1], [2]
         - Retrieved source chunks with symbols & line numbers
         - Citation validation
         - Grounding validation
         - Confidence score
         - Execution timing breakdown
    6. Switch repository at any time with '/repo <url>' or typing 'exit' at the question prompt.
    7. Exit application with '/exit' or 'exit'.

No internal paths or vector store details are ever required from the user.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.repository_rag import RepositoryRAG, EMBEDDING_MODEL
from app.repo_indexer import normalize_github_url

logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s] %(message)s",
)

SEPARATOR = "=" * 65


def check_groq_config() -> None:
    """Verify GROQ_API_KEY is configured in the environment."""
    if not os.environ.get("GROQ_API_KEY"):
        print(f"\n{SEPARATOR}")
        print("ERROR: GROQ_API_KEY is not configured.")
        print()
        print("Set GROQ_API_KEY in your environment and restart.")
        print(f"{SEPARATOR}\n")
        sys.exit(1)


def print_header() -> None:
    print()
    print(SEPARATOR)
    print("                 CODE REPOSITORY RAG")
    print(SEPARATOR)
    print("  GitHub URL → Clone → Chunk → Embed → FAISS → Groq")
    print("  Hybrid RRF Retrieval · Code Reranking")
    print("  Grounding · Citations · Confidence")
    print(SEPARATOR)


def print_ready(info: dict) -> None:
    cached_note = " (cached)" if info.get("cached") else ""
    print()
    print(SEPARATOR)
    print(f"  REPOSITORY READY{cached_note}")
    print(SEPARATOR)
    print(f"  Repository : {info.get('repository_url')}")
    print(f"  Files      : {info.get('files_indexed', 0):,}")
    print(f"  Chunks     : {info.get('chunks_indexed', 0):,}")
    print(f"  Embedding  : {info.get('embedding_model')} ({info.get('embedding_dimension')}d)")
    print(f"  Vector DB  : FAISS")
    print(f"  LLM        : {info.get('llm_model')}")
    print(SEPARATOR)


def print_result(result: dict) -> None:
    print()
    print(SEPARATOR)
    print("ANSWER")
    print(SEPARATOR)
    print(result["answer"])
    print()

    # Sources
    sources = result.get("citations", [])
    if sources:
        print("Sources:")
        for s in sources:
            symbol_part = f" | {s['symbol']}" if s.get("symbol") else ""
            lines_part = (
                f" | Lines {s['start_line']}-{s['end_line']}"
                if s.get("start_line") is not None and s.get("end_line") is not None
                else ""
            )
            print(f"[{s['number']}] {s['file']}{symbol_part}{lines_part}")
            if s.get("github"):
                print(f"     {s['github']}")
        print()

    # Validation & Confidence
    cv = result.get("citation_validation", {})
    cv_status = "PASS" if cv.get("valid") else f"FAIL (Invalid: {cv.get('invalid_citations')})"
    print(f"Citations  : {cv_status}")

    gv = result.get("grounding_validation", {})
    gv_status = "PASS" if gv.get("passed") else "FAIL"
    if result.get("repair_attempted"):
        gv_status += " (repaired)" if result.get("repair_passed") else " (repair unverified)"
    print(f"Grounding  : {gv_status}")
    if gv.get("unsupported_claims"):
        for claim in gv["unsupported_claims"]:
            print(f"  - Unsupported: {claim}")

    conf = result.get("confidence", {})
    conf_level = conf.get("level", "UNKNOWN")
    conf_score = conf.get("confidence_percent", 0.0)
    print(f"Confidence : {conf_level} ({conf_score:.1f}%)")

    timing = result.get("timing", {})
    if timing:
        print(
            f"Timing     : Retrieval: {timing.get('retrieval', 0.0):.2f}s | "
            f"Generation: {timing.get('generation', 0.0):.2f}s | "
            f"Validation: {timing.get('validation', 0.0):.2f}s | "
            f"Total: {timing.get('total', 0.0):.2f}s"
        )
    print(SEPARATOR)


def run_session(repo_url: str) -> None:
    rag = RepositoryRAG(repo_url)

    try:
        info = rag.initialize()
    except Exception as exc:
        print(f"\nERROR initializing repository: {exc}\n")
        return

    print_ready(info)

    while True:
        try:
            prompt_text = "\nAsk your question (type 'exit' to switch repo): "
            question = input(prompt_text).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nReturning to repository selection...")
            break

        if question.lower() in ("exit", "quit", ":q"):
            break

        if question.lower().startswith("/repo "):
            new_url = question[6:].strip()
            if new_url:
                rag.close()
                run_session(new_url)
                return

        if not question:
            continue

        print("\nSearching repository...")
        print("Generating answer...")

        try:
            result = rag.ask(question)
            print_result(result)
        except Exception as exc:
            print(f"\nERROR during query processing: {exc}")

    rag.close()


def main() -> None:
    print_header()
    check_groq_config()

    while True:
        try:
            url = input("\nEnter GitHub repository URL (or 'exit'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            return

        if url.lower() in ("exit", "quit", ":q", "/exit"):
            print("Goodbye.")
            return

        if not url:
            continue

        try:
            normalize_github_url(url)
        except ValueError as exc:
            print(f"\nInvalid GitHub repository URL: {exc}")
            print("Example: https://github.com/owner/repository")
            continue

        run_session(url)


if __name__ == "__main__":
    main()
