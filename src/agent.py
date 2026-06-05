from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        retrieved = self.store.search(question, top_k=top_k)
        if not retrieved:
            prompt = f"Question: {question}\n\nNo relevant context was retrieved. Answer concisely."
            return self.llm_fn(prompt)

        context_sections: list[str] = []
        for index, result in enumerate(retrieved, start=1):
            metadata = result.get("metadata") or {}
            metadata_preview = ", ".join(
                f"{key}={value}"
                for key, value in metadata.items()
                if key not in {"content", "doc_id"}
            )
            context_sections.append(
                f"Chunk {index} | score={result['score']:.4f} | {metadata_preview}\n{result['content']}"
            )

        prompt = (
            "You are a knowledge agent. Use the retrieved context to answer the question as accurately as possible.\n\n"
            "--- Retrieved Context ---\n"
            f"{chr(10).join(context_sections)}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        return self.llm_fn(prompt)
