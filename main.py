from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/python_intro.txt",
    "data/vector_store_notes.md",
    "data/rag_system_design.md",
    "data/customer_support_playbook.txt",
    "data/chunking_experiment_report.md",
    "data/vi_retrieval_notes.md",
]

FILES = [ 
    "new_data/001_luat_52656.md", 
    "new_data/002_nghi_dinh_8908.md", 
    "new_data/003_nghi_dinh_14849.md", 
    "new_data/004_nghi_dinh_124155.md", 
    "new_data/005_nghi_dinh_164307.md", 
    "new_data/006_nghi_dinh_219863.md", 
    "new_data/007_thong_tu_18000.md", 
    "new_data/008_thong_tu_304257.md", 
    "new_data/009_thong_tu_313701.md", 
    "new_data/010_thong_tu_316062.md" 
    ]

def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def chunk_documents(documents: list[Document], chunk_size: int = 3000, overlap: int = 200) -> list[Document]:
    chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    chunked_documents: list[Document] = []

    for doc in documents:
        chunks = chunker.chunk(doc.content)
        for index, chunk in enumerate(chunks, start=1):
            chunked_documents.append(
                Document(
                    id=f"{doc.id}_{index}",
                    content=chunk,
                    metadata={
                        **doc.metadata,
                        "original_doc_id": doc.id,
                        "chunk_index": index,
                    },
                )
            )
    return chunked_documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def real_llm(prompt: str) -> str:
    """Call a real OpenAI-based LLM using the configured environment."""
    try:
        from openai import OpenAI

        client = OpenAI()
        llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"[LLM Error: {exc}]")
        return demo_llm(prompt)


def get_llm_fn() -> Callable[[str], str]:
    use_real = os.getenv("LLM_PROVIDER", "openai").strip().lower() in ("openai", "default", "real")
    if use_real:
        return real_llm
    return demo_llm


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    chunk_size = int(os.getenv("CHUNK_SIZE", "3000"))
    overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    chunked_docs = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
    print(f"Converted to {len(chunked_docs)} chunked documents using chunk_size={chunk_size}, overlap={overlap}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(chunked_docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    llm_fn = get_llm_fn()
    print(f"LLM backend: {llm_fn.__name__}")
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
