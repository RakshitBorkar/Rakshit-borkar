"""
scripts/ingest_docs.py
──────────────────────
Ingest documents into the RAG vector store.

Run:
    python scripts/ingest_docs.py --source ./data/docs/
    python scripts/ingest_docs.py --url https://example.com/article
    python scripts/ingest_docs.py --text "Your custom text here" --source "manual"
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.pipeline import RAGPipeline
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG vector store")
    parser.add_argument("--source",   type=str, help="Directory or file path to ingest")
    parser.add_argument("--url",      type=str, help="URL to ingest")
    parser.add_argument("--text",     type=str, help="Raw text to ingest")
    parser.add_argument("--name",     type=str, default="manual", help="Source name for raw text")
    parser.add_argument("--store",    type=str, default="./data/vectorstore")
    parser.add_argument("--chunk",    type=int, default=512)
    parser.add_argument("--overlap",  type=int, default=64)
    args = parser.parse_args()

    rag = RAGPipeline(
        persist_dir=args.store,
        chunk_size=args.chunk,
        chunk_overlap=args.overlap,
    )

    if args.source:
        p = Path(args.source)
        if p.is_dir():
            logger.info(f"Ingesting directory: {args.source}")
            rag.ingest_directory(args.source)
        elif p.is_file():
            logger.info(f"Ingesting file: {args.source}")
            rag.ingest_file(args.source)
        else:
            logger.error(f"Path not found: {args.source}")

    elif args.url:
        logger.info(f"Ingesting URL: {args.url}")
        rag.ingest_url(args.url)

    elif args.text:
        logger.info(f"Ingesting raw text ({len(args.text)} chars)")
        rag.ingest_text(args.text, source=args.name)

    else:
        parser.print_help()
        return

    stats = rag.stats()
    logger.success(f"Done! Vector store stats: {stats}")


if __name__ == "__main__":
    main()