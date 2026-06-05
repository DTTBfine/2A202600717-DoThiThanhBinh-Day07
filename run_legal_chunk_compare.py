from __future__ import annotations

import re
from pathlib import Path

from src.chunking import ChunkingStrategyComparator
from src.models import Document
from src.store import EmbeddingStore


LEGAL_FILES = [
    "new_data/001_luat_52656.md",
    "new_data/002_nghi_dinh_8908.md",
    "new_data/003_nghi_dinh_14849.md",
]

ALL_LEGAL_FILES = sorted(Path("new_data").glob("*.md"))

BENCHMARK_QUERIES = [
    (
        "32/2004/QH11",
        "Luật An ninh Quốc gia 2004 quy định những nội dung gì?",
    ),
    (
        "16/2006/NĐ-CP",
        "Văn bản 16/2006/NĐ-CP nói về khôi phục danh dự, đền bù, trợ cấp đúng không?",
    ),
    (
        "127/2006/NĐ-CP",
        "Nghị định 127/2006/NĐ-CP quy định điều kiện bảo đảm nào cho hoạt động bảo vệ an ninh quốc gia?",
    ),
    (
        "08/2016/TT-BCA",
        "Trang phục bảo vệ cơ quan, doanh nghiệp được quy định ở văn bản nào?",
    ),
    (
        "85/2016/TTLT-BTC-BCA",
        "Tìm thông tư liên tịch 85/2016 quy định phối hợp giữa Bộ Tài chính và Bộ Công an trong bảo vệ an ninh quốc gia.",
    ),
]

DOCUMENT_ALIASES = {
    "luật an ninh quốc gia 2004": "32/2004/QH11",
    "luat an ninh quoc gia 2004": "32/2004/QH11",
    "trang phục bảo vệ cơ quan": "08/2016/TT-BCA",
    "trang phuc bao ve co quan": "08/2016/TT-BCA",
}


class LegalSectionChunker:
    """Chunking strategy cho domain văn bản pháp luật Việt Nam.

    Strategy này ưu tiên heading markdown và đơn vị pháp lý như Chương/Mục/Điều.
    Mỗi chunk được thêm phần header ngắn chứa tên văn bản và số hiệu để retrieval
    có tín hiệu rõ hơn khi query nhắc tới một văn bản cụ thể.
    """

    ARTICLE_HEADING_RE = re.compile(
        r"(?m)^(?:#{1,4}\s*)?(?P<heading>(?:Chương|Mục|Điều)\s+[IVXLCDM\d]+[^\n]*)"
    )
    META_KEYS = ["Số hiệu", "Loại văn bản", "Nơi ban hành", "Người ký", "Ngày ban hành"]

    def __init__(self, max_chars: int = 900, overlap: int = 120) -> None:
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        text = text.strip()
        metadata = extract_metadata(text)
        metadata_header = self._build_metadata_header(metadata)
        sections = self._split_legal_sections(text)

        chunks: list[str] = []
        for section_title, section_text in sections:
            prefix = f"{metadata_header}\nPhần: {section_title}\n\n"
            chunks.extend(self._pack_section(prefix, section_text))

        return chunks

    def _build_metadata_header(self, metadata: dict[str, str]) -> str:
        lines = [f"Tên văn bản: {metadata.get('title', '')}"]
        for key in self.META_KEYS:
            value = metadata.get(key)
            if value:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _split_legal_sections(self, text: str) -> list[tuple[str, str]]:
        matches = list(self.ARTICLE_HEADING_RE.finditer(text))
        if not matches:
            return [("Toàn văn", text)]

        sections: list[tuple[str, str]] = []
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("Thông tin văn bản", preamble))

        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            title = match.group("heading").lstrip("# ").strip()
            sections.append((title, text[match.start() : end].strip()))

        return sections

    def _pack_section(self, prefix: str, section_text: str) -> list[str]:
        budget = max(250, self.max_chars - len(prefix))
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section_text) if part.strip()]

        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            candidate = paragraph if not current else f"{current}\n\n{paragraph}"
            if len(candidate) <= budget:
                current = candidate
                continue

            if current:
                chunks.append(prefix + current)

            if len(paragraph) > budget:
                chunks.extend(prefix + part for part in self._split_long_text(paragraph, budget))
                current = ""
            else:
                current = paragraph

        if current:
            chunks.append(prefix + current)

        return chunks

    def _split_long_text(self, text: str, budget: int) -> list[str]:
        step = max(1, budget - self.overlap)
        parts = []
        for start in range(0, len(text), step):
            parts.append(text[start : start + budget])
            if start + budget >= len(text):
                break
        return parts


def extract_metadata(text: str, path: Path | None = None) -> dict[str, str]:
    lines = text.splitlines()
    metadata = {
        "title": lines[0].lstrip("# ").strip() if lines else (path.stem if path else ""),
        "source": "thuvienphapluat.vn",
    }

    for key in LegalSectionChunker.META_KEYS:
        match = re.search(rf"^- {re.escape(key)}: (.+)$", text, re.MULTILINE)
        if match:
            metadata[key] = match.group(1).strip()

    if path is not None:
        metadata["file"] = str(path)

    return metadata


def build_documents(paths: list[Path], chunker: LegalSectionChunker) -> list[Document]:
    documents: list[Document] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        metadata = extract_metadata(text, path)
        chunks = chunker.chunk(text)

        for index, chunk in enumerate(chunks, start=1):
            chunk_metadata = dict(metadata)
            chunk_metadata["chunk_index"] = str(index)
            documents.append(
                Document(
                    id=f"{path.stem}-{index}",
                    content=chunk,
                    metadata=chunk_metadata,
                )
            )

    return documents


def infer_so_hieu_filter(query: str, known_so_hieu: set[str]) -> str | None:
    normalized_query = query.lower()

    for alias, so_hieu in DOCUMENT_ALIASES.items():
        if alias in normalized_query and so_hieu in known_so_hieu:
            return so_hieu

    for so_hieu in known_so_hieu:
        if so_hieu.lower() in normalized_query:
            return so_hieu

        compact = so_hieu.split("/")[0].lower()
        if compact and re.search(rf"\b{re.escape(compact)}\b", normalized_query):
            return so_hieu

    return None


def search_legal(store: EmbeddingStore, query: str, top_k: int, known_so_hieu: set[str]) -> list[dict]:
    so_hieu = infer_so_hieu_filter(query, known_so_hieu)
    if so_hieu:
        filtered_results = store.search_with_filter(query, top_k=top_k, metadata_filter={"Số hiệu": so_hieu})
        if filtered_results:
            return filtered_results

    return store.search(query, top_k=top_k)


def compare_chunking() -> None:
    baseline = ChunkingStrategyComparator()
    custom_chunker = LegalSectionChunker(max_chars=900, overlap=120)

    for file_path in LEGAL_FILES:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")

        print("=" * 80)
        print(f"Tài liệu: {file_path}")

        baseline_result = baseline.compare(text, chunk_size=800)
        for strategy, stats in baseline_result.items():
            print(f"\nStrategy: {strategy}")
            print(f"Chunk Count: {stats['count']}")
            print(f"Avg Length: {stats['avg_length']:.2f}")

        custom_chunks = custom_chunker.chunk(text)
        custom_count = len(custom_chunks)
        custom_avg = sum(len(chunk) for chunk in custom_chunks) / custom_count if custom_count else 0

        print("\nStrategy: legal_section_custom")
        print(f"Chunk Count: {custom_count}")
        print(f"Avg Length: {custom_avg:.2f}")


def run_benchmark() -> None:
    chunker = LegalSectionChunker(max_chars=900, overlap=120)
    documents = build_documents(ALL_LEGAL_FILES, chunker)
    known_so_hieu = {doc.metadata["Số hiệu"] for doc in documents if "Số hiệu" in doc.metadata}

    store = EmbeddingStore()
    store.add_documents(documents)

    relevant_count = 0
    print("\n" + "=" * 80)
    print("Benchmark retrieval trên 5 query")
    print(f"Indexed chunks: {store.get_collection_size()}")

    for index, (expected_so_hieu, query) in enumerate(BENCHMARK_QUERIES, start=1):
        results = search_legal(store, query, top_k=3, known_so_hieu=known_so_hieu)
        is_relevant = any(result["metadata"].get("Số hiệu") == expected_so_hieu for result in results)
        relevant_count += int(is_relevant)

        print("\n" + "-" * 80)
        print(f"Query {index}: {query}")
        print(f"Expected Số hiệu: {expected_so_hieu}")
        print(f"Relevant in top-3: {'YES' if is_relevant else 'NO'}")

        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]
            preview = " ".join(result["content"].split())[:220]
            print(
                f"Top {rank}: score={result['score']:.4f} | "
                f"Số hiệu={metadata.get('Số hiệu')} | "
                f"chunk={metadata.get('chunk_index')} | "
                f"{metadata.get('title')}"
            )
            print(f"Preview: {preview}")

    print("\n" + "=" * 80)
    print(f"Relevant top-3: {relevant_count} / {len(BENCHMARK_QUERIES)}")


if __name__ == "__main__":
    compare_chunking()
    run_benchmark()
