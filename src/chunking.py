from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        text = text.strip()
        if not text:
            return []

        sentence_splitter = re.compile(r"(?<=[.!?])\s+|\n+")
        sentences = [sentence.strip() for sentence in sentence_splitter.split(text) if sentence.strip()]

        chunks: list[str] = []
        current_chunk: list[str] = []

        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= self.max_sentences_per_chunk:
                chunks.append(" ".join(current_chunk).strip())
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())

        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []

        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        separator = remaining_separators[0]
        if separator == "":
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        pieces = current_text.split(separator)
        if len(pieces) == 1:
            return self._split(current_text, remaining_separators[1:])

        chunks: list[str] = []
        current_piece = ""

        for piece in pieces:
            candidate = piece if not current_piece else f"{current_piece}{separator}{piece}"
            if len(candidate) <= self.chunk_size:
                current_piece = candidate
            else:
                if current_piece:
                    chunks.extend(self._split(current_piece, remaining_separators[1:]))
                current_piece = piece

        if current_piece:
            chunks.extend(self._split(current_piece, remaining_separators[1:]))

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0

    dot_product = _dot(vec_a, vec_b)
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed_chunks = FixedSizeChunker(chunk_size=chunk_size, overlap=50 if chunk_size > 50 else 0).chunk(text)
        sentence_chunks = SentenceChunker(max_sentences_per_chunk=3).chunk(text)
        recursive_chunks = RecursiveChunker(chunk_size=chunk_size).chunk(text)

        def build_stats(chunks: list[str]) -> dict:
            count = len(chunks)
            avg_length = sum(len(chunk) for chunk in chunks) / count if count else 0.0
            return {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks,
            }

        return {
            "fixed_size": build_stats(fixed_chunks),
            "by_sentences": build_stats(sentence_chunks),
            "recursive": build_stats(recursive_chunks),
        }
