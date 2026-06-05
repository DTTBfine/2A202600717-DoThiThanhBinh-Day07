# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Đỗ Thị Thanh Bình  
**Nhóm:** Nhóm C7  
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector embedding có hướng gần giống nhau, tức là hai câu hoặc đoạn văn có mức độ tương đồng cao về mặt ngữ nghĩa. Giá trị càng gần 1 thì nội dung càng giống nhau.

**Ví dụ HIGH similarity:**
- Sentence A: Khách hàng muốn đổi trả sản phẩm bị lỗi.
- Sentence B: Người mua yêu cầu hoàn tiền vì sản phẩm bị hỏng.
- Tại sao tương đồng: Hai câu đều nói về việc khách hàng gặp vấn đề với sản phẩm lỗi/hỏng và muốn xử lý bằng đổi trả hoặc hoàn tiền.

**Ví dụ LOW similarity:**
- Sentence A: Khách hàng muốn đổi trả sản phẩm bị lỗi.
- Sentence B: Hôm nay thời tiết nắng đẹp và ít mây.
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau: một câu nói về đổi trả sản phẩm, câu còn lại nói về thời tiết.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity tập trung vào hướng của vector thay vì độ lớn tuyệt đối, nên phù hợp hơn để đo mức độ giống nhau về ngữ nghĩa giữa các text embeddings. Trong NLP, hai câu có thể có độ dài hoặc magnitude khác nhau nhưng vẫn mang ý nghĩa tương tự nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Trình bày phép tính:  
> Theo công thức trong bài:
>
> `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
>
> Với `doc_length = 10000`, `chunk_size = 500`, `overlap = 50`:
>
> `step = chunk_size - overlap = 500 - 50 = 450`
>
> `num_chunks = ceil((10000 - 50) / 450)`
>
> `num_chunks = ceil(9950 / 450)`
>
> `num_chunks = ceil(22.11) = 23`

> Đáp án: **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap tăng lên 100, bước nhảy giảm còn:
>
> `step = 500 - 100 = 400`
>
> `num_chunks = ceil((10000 - 100) / 400)`
>
> `num_chunks = ceil(9900 / 400)`
>
> `num_chunks = ceil(24.75) = 25`
>
> Vậy số chunk tăng từ **23 lên 25 chunks**. Overlap nhiều hơn giúp giữ ngữ cảnh giữa các chunk liền kề, giảm nguy cơ một ý quan trọng bị cắt rời ở ranh giới chunk, nhưng cũng làm tăng số lượng chunk cần lưu và truy vấn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Vietnamese law — luật an ninh quốc gia và văn bản hướng dẫn bảo vệ an ninh quốc gia.

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain luật an ninh quốc gia vì đây là một tập tài liệu chuyên ngành, có cấu trúc rõ ràng và nhiều khái niệm pháp lý cần tra cứu chính xác. Việc sử dụng tài liệu luật giúp kiểm tra hiệu quả của RAG trong một ngữ cảnh thực tế, nơi mỗi câu hỏi đòi hỏi trích xuất đúng văn bản và xác định nhanh các điều khoản liên quan.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Luật An ninh Quốc gia 2004 số 32/2004/QH11 | https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Luat-An-ninh-Quoc-gia-2004-32-2004-QH11-52656.aspx | 26.525 | Số hiệu, Loại văn bản, Nơi ban hành, Người ký, Ngày ban hành, Nguồn |
| 2 | Nghị định 16/2006/NĐ-CP khôi phục danh dự, đền bù, trợ cấp | https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Nghi-dinh-16-2006-ND-CP-khoi-phuc-danh-du-den-bu-tro-cap-cho-co-quan-to-chuc-ca-nhan-bi-thiet-hai-do-tham-gia-bao-ve-an-ninh-quoc-gia-8908.aspx | 23.631 | Số hiệu, Loại văn bản, Nơi ban hành, Người ký, Ngày ban hành, Nguồn |
| 3 | Nghị định 127/2006/NĐ-CP bảo đảm điều kiện bảo vệ an ninh | https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Nghi-dinh-127-2006-ND-CP-bao-dam-dieu-kien-cho-bao-ve-an-ninh-quoc-gia-giu-gin-trat-tu-an-toan-xa-hoi-14849.aspx | 19.728 | Số hiệu, Loại văn bản, Nơi ban hành, Người ký, Ngày ban hành, Nguồn |
| 4 | Nghị định 35/2011/NĐ-CP biện pháp pháp luật bảo vệ an ninh | https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Nghi-dinh-35-2011-ND-CP-bien-phap-phap-luat-bao-ve-an-ninh-quoc-gia-124155.aspx | 10.866 | Số hiệu, Loại văn bản, Nơi ban hành, Người ký, Ngày ban hành, Nguồn |
| 5 | Nghị định 06/2013/NĐ-CP bảo vệ cơ quan, doanh nghiệp | https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Nghi-dinh-06-2013-ND-CP-bao-ve-co-quan-doanh-nghiep-164307.aspx | 14.061 | Số hiệu, Loại văn bản, Nơi ban hành, Người ký, Ngày ban hành, Nguồn |
| 6 | Nghị định 06/2014/NĐ-CP vận động quần chúng bảo vệ an ninh quốc gia | https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Nghi-dinh-06-2014-ND-CP-van-dong-quan-chung-bao-ve-an-ninh-quoc-gia-giu-gin-trat-tu-xa-hoi-219863.aspx | 10.121 | Số hiệu, Loại văn bản, Nơi ban hành, Người ký, Ngày ban hành, Nguồn |
| 7 | Thông tư liên tịch 02/2007/TTLT-BCA-BLĐTBXH-BTC hướng dẫn Nghị định 38/2006 | https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Thong-tu-lien-tich-02-2007-TTLT-BCA-BLDTBXH-BTC-huongg-dan-Nghi-dinh-38-2006-ND-CP-Bao-ve-dan-pho-18000.aspx | 29.500 | Số hiệu, Loại văn bản, Nơi ban hành, Người ký, Ngày ban hành, Nguồn |
| 8 | Thông tư 08/2016/TT-BCA trang phục bảo vệ cơ quan, doanh nghiệp | https://thuvienphapluat.vn/van-ban/Doanh-nghiep/Thong-tu-08-2016-TT-BCA-trang-phuc-bao-ve-co-quan-doanh-nghiep-304257.aspx | 17.812 | Số hiệu, Loại văn bản, Nơi ban hành, Người ký, Ngày ban hành, Nguồn |
| 9 | Thông tư 02/2016/TTLT-BVHTTDL-BCA bảo vệ an ninh quốc gia văn hóa gia đình | https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Thong-tu-02-2016-TTLT-BVHTTDL-BCA-bao-ve-an-ninh-quoc-gia-trat-tu-an-toan-xa-hoi-van-hoa-gia-dinh-313701.aspx | 21.774 | Số hiệu, Loại văn bản, Nơi ban hành, Người ký, Ngày ban hành, Nguồn |
| 10 | Thông tư liên tịch 85/2016/TTLT-BTC-BCA phối hợp công tác bảo vệ an ninh quốc gia | https://thuvienphapluat.vn/van-ban/Tai-chinh-nha-nuoc/Thong-tu-lien-tich-85-2016-TTLT-BTC-BCA-phoi-hop-cong-tac-bao-ve-an-ninh-quoc-gia-linh-vuc-tai-chinh-316062.aspx | 35.365 | Số hiệu, Loại văn bản, Nơi ban hành, Người ký, Ngày ban hành, Nguồn |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| Số hiệu | Chuỗi | 32/2004/QH11, 16/2006/NĐ-CP | Giúp lọc và xác định chính xác văn bản pháp luật khi truy vấn theo số hiệu hoặc văn bản cụ thể |
| Loại văn bản | Chuỗi | Luật, Nghị định, Thông tư, Thông tư liên tịch | Giúp phân biệt cấp độ pháp lý và giới hạn tập dữ liệu phù hợp cho truy vấn luật pháp |
| Nơi ban hành | Chuỗi | Quốc hội, Chính phủ, Bộ Công An | Hỗ trợ truy vấn theo cơ quan ban hành, hữu ích khi tìm văn bản theo nguồn quyền lực |
| Người ký / Cơ quan ký | Chuỗi | Nguyễn Văn An, Nguyễn Tấn Dũng, Bộ Công An | Cho phép truy vấn gián tiếp theo tác giả hoặc cơ quan chịu trách nhiệm ban hành văn bản |
| Ngày ban hành | Ngày | 03/12/2004, 25/01/2006 | Giúp sắp xếp và lọc các văn bản theo thời gian ban hành |
| Nguồn | Chuỗi | https://thuvienphapluat.vn/... | Giúp xác thực và đối chiếu nguồn dữ liệu khi cần so sánh hoặc kiểm tra tính chính xác |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu đại diện, với `chunk_size=800`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Luật An ninh Quốc gia 2004 | FixedSizeChunker (`fixed_size`) | 36 | 785.42 | Trung bình — chunk dài và gần sát `chunk_size`, có nhiều ngữ cảnh hơn nhưng vẫn có thể cắt ngang câu, điều hoặc khoản |
| Luật An ninh Quốc gia 2004 | SentenceChunker (`by_sentences`) | 107 | 245.08 | Khá tốt — giữ được ranh giới câu, chunk dài hơn và dễ đọc hơn, nhưng có thể gom nhiều câu khiến một số chunk vượt quá `chunk_size` |
| Luật An ninh Quốc gia 2004 | RecursiveChunker (`recursive`) | 38 | 696.05 | Khá tốt — ưu tiên tách theo đoạn/dòng/câu nên giữ ngữ cảnh tốt hơn fixed-size, nhưng vẫn chưa nhận diện trực tiếp số điều/khoản |
| Nghị định 16/2006/NĐ-CP | FixedSizeChunker (`fixed_size`) | 32 | 786.91 | Trung bình — giữ độ dài chunk rất đều và nhiều ngữ cảnh hơn, nhưng không nhận biết cấu trúc pháp lý như Điều, Khoản, Mục |
| Nghị định 16/2006/NĐ-CP | SentenceChunker (`by_sentences`) | 78 | 300.21 | Khá tốt — số chunk ít hơn và giữ câu hoàn chỉnh, phù hợp khi câu pháp lý chứa đủ ý, nhưng chunk có thể khá dài |
| Nghị định 16/2006/NĐ-CP | RecursiveChunker (`recursive`) | 38 | 619.89 | Khá tốt — chia theo separator tự nhiên nên chunk dễ đọc hơn fixed-size, nhưng có thể tách rời tiêu đề với nội dung điều |
| Nghị định 127/2006/NĐ-CP | FixedSizeChunker (`fixed_size`) | 27 | 778.81 | Trung bình — kích thước đều và nhiều ngữ cảnh, nhưng có rủi ro cắt ngang nội dung pháp lý quan trọng |
| Nghị định 127/2006/NĐ-CP | SentenceChunker (`by_sentences`) | 65 | 300.72 | Tốt — ít chunk hơn, giữ được câu hoàn chỉnh và nhiều ngữ cảnh hơn trong mỗi chunk |
| Nghị định 127/2006/NĐ-CP | RecursiveChunker (`recursive`) | 30 | 655.63 | Khá tốt — tách linh hoạt theo separator, giữ ngữ cảnh khá đầy đủ nhưng chưa tận dụng pattern pháp luật như `Điều` |

### Strategy Của Tôi

**Loại:** Custom strategy — `LegalSectionChunker`

**Mô tả cách hoạt động:**
> Strategy của tôi chia văn bản theo cấu trúc pháp luật, ưu tiên heading markdown và các đơn vị như `Chương`, `Mục`, `Điều`. Mỗi chunk được thêm header ngắn gồm tên văn bản, số hiệu, loại văn bản, nơi ban hành, người ký và ngày ban hành. Sau đó thuật toán pack các đoạn trong cùng một điều/mục đến `max_chars=900`; nếu một đoạn quá dài thì mới cắt nhỏ với overlap 120 ký tự. Cách này vừa giữ đơn vị pháp lý có nghĩa, vừa giúp retrieval nhận được tín hiệu metadata khi query nhắc số hiệu văn bản.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu pháp luật thường có cấu trúc rất rõ theo chương, mục, điều, khoản. Người dùng thường hỏi theo một quy định hoặc một văn bản cụ thể, nên giữ trọn nội dung quanh `Điều` hoặc `Mục` sẽ giúp retrieval chính xác và dễ grounding hơn so với các strategy chia nhỏ theo ký tự hoặc theo từng cụm câu ngắn.

**Code snippet (nếu custom):**
```python
import re
from pathlib import Path

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
```

### So Sánh: Strategy của tôi vs Baseline

Số liệu dưới đây lấy từ lần chạy thật `python3 run_legal_chunk_compare.py`.

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Luật An ninh Quốc gia 2004 | best baseline: RecursiveChunker | 38 | 696.05 | Khá tốt — giữ được nhiều ngữ cảnh hơn SentenceChunker và tránh cắt quá thô như fixed-size, nhưng chưa thêm metadata vào từng chunk |
| Luật An ninh Quốc gia 2004 | **của tôi: LegalSectionChunker** | 65 | 668.00 | Tốt hơn cho truy vấn pháp luật vì giữ được cụm nội dung quanh Chương/Mục/Điều và thêm header metadata cho từng chunk |
| Nghị định 16/2006/NĐ-CP | best baseline: RecursiveChunker | 38 | 619.89 | Khá tốt — chunk đủ dài và tách theo separator tự nhiên, nhưng chưa tận dụng trực tiếp cấu trúc Điều/Khoản và số hiệu văn bản |
| Nghị định 16/2006/NĐ-CP | **của tôi: LegalSectionChunker** | 69 | 693.36 | Tốt hơn khi query cần số hiệu văn bản và ngữ cảnh đầy đủ về điều kiện, trách nhiệm, trình tự hoặc đối tượng áp dụng |
| Nghị định 127/2006/NĐ-CP | best baseline: RecursiveChunker | 30 | 655.63 | Khá tốt — ít chunk và giữ ngữ cảnh khá đầy đủ, nhưng chưa nhận diện pattern pháp luật như `Điều` và chưa thêm metadata header |
| Nghị định 127/2006/NĐ-CP | **của tôi: LegalSectionChunker** | 57 | 695.60 | Tốt hơn cho câu hỏi theo điều khoản pháp lý vì chunk vừa đủ dài, có metadata và ít bị cắt ngang đơn vị pháp lý |

**Đánh giá:**
> Kết quả cho thấy `LegalSectionChunker` tạo số chunk cân bằng hơn baseline: 65, 69 và 57 chunk trên ba tài liệu. Avg Length khoảng 668–696 ký tự, đủ giữ ngữ cảnh pháp lý nhưng không quá dài. Việc thêm header metadata vào từng chunk giúp các query có số hiệu văn bản retrieval ổn định hơn.

> So với `RecursiveChunker`, custom strategy tạo số chunk nhiều hơn một chút nhưng bám sát đơn vị pháp lý hơn và có metadata rõ trong từng chunk. Vì vậy, tôi đánh giá `LegalSectionChunker` phù hợp hơn cho bộ tài liệu nhóm.

### Kiểm Tra Retrieval Với Strategy Của Tôi

Chạy `python3 run_legal_chunk_compare.py` trên 10 tài liệu trong `new_data/`, dùng `LegalSectionChunker(max_chars=900, overlap=120)` và metadata-aware retrieval theo `Số hiệu`.

| # | Query benchmark | Top-1 Số hiệu | Top-1 Score | Relevant trong top-3? |
|---|-----------------|---------------|-------------|------------------------|
| 1 | Luật An ninh Quốc gia 2004 quy định những nội dung gì? | 32/2004/QH11 | 0.3469 | Có |
| 2 | Văn bản 16/2006/NĐ-CP nói về khôi phục danh dự, đền bù, trợ cấp đúng không? | 16/2006/NĐ-CP | 0.3296 | Có |
| 3 | Nghị định 127/2006/NĐ-CP quy định điều kiện bảo đảm nào cho hoạt động bảo vệ an ninh quốc gia? | 127/2006/NĐ-CP | 0.2934 | Có |
| 4 | Trang phục bảo vệ cơ quan, doanh nghiệp được quy định ở văn bản nào? | 08/2016/TT-BCA | 0.3393 | Có |
| 5 | Tìm thông tư liên tịch 85/2016 quy định phối hợp giữa Bộ Tài chính và Bộ Công an trong bảo vệ an ninh quốc gia. | 85/2016/TTLT-BTC-BCA | 0.2582 | Có |

**Tổng số custom chunks đã index:** **566**  
**Relevant top-3:** **5 / 5**

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | LegalSectionChunker — chunk theo Chương/Mục/Điều, thêm metadata header và filter theo số hiệu | 10/10 | Giữ tốt ngữ cảnh pháp lý, top-3 relevant 5/5 trên benchmark | Cần metadata tốt; nếu không có số hiệu/alias thì mock embedding vẫn có thể nhiễu |
| Lê Vũ Anh | RecursiveChunker | 8/10 | Giữ cấu trúc đoạn, dòng, câu; phù hợp văn bản dài | Có thể tạo chunk dài, cần chọn chunk size hợp lý, chưa tận dụng trực tiếp pattern pháp luật như `Điều`, `Khoản` |
| Lê Trung Kiên | SentenceChunker | 8/10 | Giữ nguyên câu, dễ đọc hơn fixed-size | Chưa tận dụng trực tiếp pattern pháp luật như `Điều`, `Khoản` hay xuống dòng |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Với domain văn bản pháp luật, strategy tốt nhất là chunk theo cấu trúc pháp lý như `Chương`, `Mục`, `Điều`, vì đây là cách tài liệu được tổ chức tự nhiên. Strategy này giúp retrieved chunk chứa đủ tiêu đề và nội dung liên quan, từ đó agent dễ trả lời có căn cứ hơn và giảm nguy cơ hallucination.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Tôi dùng regex `(?<=[.!?])\s+|\n+` để tách câu dựa trên dấu kết thúc câu như `.`, `!`, `?` hoặc xuống dòng. Sau khi tách, tôi `strip()` để loại bỏ khoảng trắng thừa và bỏ các câu rỗng. Các câu được gom theo `max_sentences_per_chunk`, nếu cuối văn bản còn câu chưa đủ một chunk thì vẫn được thêm vào kết quả.

**`RecursiveChunker.chunk` / `_split`** — approach:
> RecursiveChunker hoạt động bằng cách thử tách văn bản theo thứ tự separator ưu tiên: đoạn văn, dòng mới, câu, khoảng trắng, rồi cuối cùng mới cắt cứng theo ký tự. Base case là khi đoạn hiện tại rỗng hoặc độ dài nhỏ hơn/equal `chunk_size` thì trả về luôn. Nếu một separator không tách được văn bản, thuật toán chuyển sang separator tiếp theo; nếu vẫn quá dài thì tiếp tục gọi `_split` đệ quy trên phần đó.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Trong `add_documents`, mỗi `Document` được chuyển thành một record gồm `id`, `doc_id`, `content`, `metadata` và `embedding`. Embedding được tạo bằng `embedding_fn`, mặc định là mock embedder để test không cần API key. Trong `search`, nếu ChromaDB chưa khả dụng thì store dùng in-memory records: query cũng được embed, sau đó `_search_records` tính điểm bằng dot product giữa query embedding và document embedding, rồi sort giảm dần để lấy top-k. Vì mock embedding đã được normalize nên dot product có thể xem gần tương đương cosine trong lần chạy mặc định này.

**`search_with_filter` + `delete_document`** — approach:
> Với `search_with_filter`, tôi filter metadata trước để thu hẹp tập candidate, sau đó mới gọi `_search_records` trên các record phù hợp. Cách này giúp query như “chỉ tìm trong Nghị định” hoặc “chỉ tìm trong một số hiệu văn bản cụ thể” chính xác hơn nếu metadata đã được gán tốt. Với `delete_document`, tôi xóa tất cả record có `doc_id` trùng với document cần xóa và trả về `True` nếu kích thước store giảm.

### KnowledgeBaseAgent

**`answer`** — approach:
> Agent dùng RAG pattern: trước tiên gọi `store.search(question, top_k)` để lấy các chunk liên quan, sau đó build prompt gồm retrieved context, score, metadata và nội dung chunk. Context được inject vào phần `--- Retrieved Context ---`, tiếp theo là câu hỏi của người dùng và nhãn `Answer:`. Nếu không retrieve được chunk nào, agent vẫn gọi LLM với prompt báo rằng không có context liên quan để trả lời ngắn gọn.

### Test Results

```bash
$ PYTHONPATH=. pytest tests/ -v
=========================================== test session starts ===========================================
platform darwin -- Python 3.11.14, pytest-9.0.3, pluggy-1.6.0 -- /Users/dttbfine/Downloads/AI20K/Day-07-Lab-Data-Foundations/.venv/bin/python3.11
cachedir: .pytest_cache
rootdir: /Users/dttbfine/Downloads/AI20K/Day-07-Lab-Data-Foundations
plugins: anyio-4.13.0
collected 42 items                                                                                        

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED               [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                        [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                 [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                  [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                       [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED       [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED             [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED              [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED            [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                              [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED              [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                         [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                     [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                               [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED      [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED          [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED    [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED          [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                              [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                  [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                        [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED             [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED               [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED   [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                         [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                        [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                   [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED               [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED          [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED              [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                    [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED              [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED         [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED        [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED       [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED[100%]

=========================================== 42 passed in 0.04s ============================================

```

**Số tests pass:** **42 / 42**

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Sử dụng `compute_similarity()` với `_mock_embed` để tạo embedding cho từng câu trước khi tính cosine similarity.

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Cơ quan có trách nhiệm bảo vệ an ninh quốc gia. | Tổ chức phải thực hiện nhiệm vụ bảo vệ an ninh quốc gia. | high | -0.0892 | Sai |
| 2 | Văn bản quy định về kinh phí bảo vệ an ninh. | Thông tư hướng dẫn cơ chế phối hợp tài chính. | high | -0.2255 | Sai |
| 3 | Công dân có nghĩa vụ tuân thủ pháp luật. | Hôm nay trời mưa lớn ở Hà Nội. | low | -0.0185 | Đúng tương đối |
| 4 | Nghị định quy định điều kiện bảo đảm an ninh. | Luật này nêu nguyên tắc bảo vệ an ninh quốc gia. | high | 0.0083 | Sai |
| 5 | Trang phục bảo vệ cơ quan doanh nghiệp được hướng dẫn trong thông tư. | Python là ngôn ngữ lập trình phổ biến cho AI. | low | 0.0214 | Đúng tương đối |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là Pair 1 và Pair 2, vì hai cặp này tương đồng về ngữ nghĩa nhưng score lại thấp hoặc âm. Nguyên nhân là lần chạy này dùng `_mock_embed`, embedding được sinh theo cách deterministic để phục vụ test chứ không thật sự hiểu nghĩa tiếng Việt. Điều này cho thấy muốn đánh giá semantic similarity nghiêm túc thì nên dùng embedding model thật như `all-MiniLM-L6-v2` hoặc OpenAI embeddings thay vì mock embeddings.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Luật An ninh Quốc gia 2004 quy định những nội dung gì? | Luật quy định về bảo vệ an ninh quốc gia, bao gồm chính sách, nguyên tắc, nhiệm vụ, biện pháp và trách nhiệm của cơ quan, tổ chức, công dân. |
| 2 | Văn bản 16/2006/NĐ-CP nói về khôi phục danh dự, đền bù, trợ cấp đúng không? | Có. Nghị định 16/2006/NĐ-CP liên quan đến khôi phục danh dự, đền bù, trợ cấp đối với cơ quan, tổ chức, cá nhân tham gia bảo vệ an ninh quốc gia nhưng bị thiệt hại hoặc ảnh hưởng quyền lợi hợp pháp. |
| 3 | Nghị định 127/2006/NĐ-CP quy định điều kiện bảo đảm nào cho hoạt động bảo vệ an ninh quốc gia? | Nghị định này quy định các điều kiện bảo đảm như kinh phí, phương tiện, cơ sở vật chất, trang thiết bị và trách nhiệm tổ chức thực hiện. |
| 4 | Trang phục bảo vệ cơ quan, doanh nghiệp được quy định ở văn bản nào? | Nội dung này được quy định trong Thông tư 08/2016/TT-BCA về trang phục, cấp hiệu, phù hiệu, biển hiệu và giấy chứng nhận của lực lượng bảo vệ cơ quan, doanh nghiệp. |
| 5 | Tìm thông tư liên tịch 85/2016 quy định phối hợp giữa Bộ Tài chính và Bộ Công an trong bảo vệ an ninh quốc gia. | Văn bản phù hợp là Thông tư liên tịch 85/2016/TTLT-BTC-BCA, quy định việc phối hợp giữa Bộ Tài chính và Bộ Công an trong công tác bảo vệ an ninh quốc gia. |

### Kết Quả Của Tôi

**Cách chạy thật:** index 10 file trong `new_data/` thành **566 chunks** bằng `LegalSectionChunker(max_chars=900, overlap=120)`, sau đó dùng `EmbeddingStore` trong `src` để search. Khi query có số hiệu hoặc alias rõ như “Luật An ninh Quốc gia 2004”, script gọi `search_with_filter()` theo metadata `Số hiệu` trước rồi mới search trong các chunk phù hợp. Cách này phù hợp hơn với domain pháp luật vì số hiệu văn bản là tín hiệu định danh chính xác.

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Answer grounded từ context (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Luật An ninh Quốc gia 2004 quy định những nội dung gì? | Chunk từ Luật An ninh Quốc gia 2004, Điều 1 về phạm vi điều chỉnh | 0.3469 | Có | Luật quy định chính sách, nguyên tắc, nhiệm vụ, biện pháp và trách nhiệm bảo vệ an ninh quốc gia |
| 2 | Văn bản 16/2006/NĐ-CP nói về khôi phục danh dự, đền bù, trợ cấp đúng không? | Chunk từ Nghị định 16/2006/NĐ-CP về khôi phục danh dự, đền bù, trợ cấp | 0.3296 | Có | Có, văn bản này quy định khôi phục danh dự, đền bù, trợ cấp cho đối tượng bị thiệt hại do tham gia bảo vệ an ninh quốc gia |
| 3 | Nghị định 127/2006/NĐ-CP quy định điều kiện bảo đảm nào cho hoạt động bảo vệ an ninh quốc gia? | Chunk từ Nghị định 127/2006/NĐ-CP về bảo đảm điều kiện cho hoạt động bảo vệ an ninh quốc gia | 0.2934 | Có | Nghị định quy định các điều kiện bảo đảm như ngân sách, cơ sở vật chất, trang thiết bị, phương tiện và trách nhiệm tổ chức thực hiện |
| 4 | Trang phục bảo vệ cơ quan, doanh nghiệp được quy định ở văn bản nào? | Chunk từ Thông tư 08/2016/TT-BCA về trang phục bảo vệ cơ quan, doanh nghiệp | 0.3393 | Có | Nội dung này được quy định trong Thông tư 08/2016/TT-BCA |
| 5 | Tìm thông tư liên tịch 85/2016 quy định phối hợp giữa Bộ Tài chính và Bộ Công an trong bảo vệ an ninh quốc gia. | Chunk từ Thông tư liên tịch 85/2016/TTLT-BTC-BCA về phối hợp trong lĩnh vực tài chính | 0.2582 | Có | Văn bản phù hợp là Thông tư liên tịch 85/2016/TTLT-BTC-BCA |

**Bao nhiêu queries trả về chunk relevant trong top-3?** **5 / 5**

**Nhận xét kết quả:**
> Sau khi cải thiện chunking và thêm metadata header vào từng chunk, retrieval ổn định hơn. Điểm quan trọng nhất là không chỉ dựa vào `_mock_embed`, vì mock embedding không hiểu tiếng Việt; với văn bản pháp luật, số hiệu như `16/2006/NĐ-CP`, `127/2006/NĐ-CP`, `08/2016/TT-BCA` là tín hiệu định danh nên cần dùng metadata filtering trước. Kết quả top-3 đạt **5/5**, và trong lần chạy này top-1 của cả 5 query đều đúng văn bản.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được rằng fixed-size chunking không phải lúc nào cũng kém, vì nếu chọn `chunk_size` và `overlap` hợp lý thì nó vẫn tạo baseline ổn định, dễ debug và dễ so sánh. Tuy nhiên, với văn bản pháp luật, việc thêm metadata như `loại văn bản`, `số hiệu`, `ngày ban hành` giúp retrieval chính xác hơn rất nhiều so với chỉ dựa vào nội dung text.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Tôi học được cách đánh giá retrieval không chỉ dựa vào việc agent có trả lời được hay không, mà còn phải xem top-3 chunk có thật sự liên quan, câu trả lời có grounded vào context hay không và metadata filtering có giúp giảm nhiễu không. Một số nhóm cũng cho thấy domain càng có cấu trúc rõ thì custom chunking càng có lợi.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Nếu làm lại, tôi sẽ chuẩn hóa metadata ngay từ đầu bằng một file `metadata.json` riêng thay vì chỉ để metadata trong tên file hoặc nhập thủ công. Tôi cũng sẽ tách văn bản theo `Điều`/`Khoản` rõ ràng hơn, đồng thời lưu thêm metadata cấp chunk như `chapter`, `article_number`, `article_title` để hỗ trợ filter chính xác. Ngoài ra, tôi sẽ dùng embedding model thật cho tiếng Việt để đánh giá similarity đáng tin cậy hơn mock embeddings.

**Failure case đã quan sát:**
> Failure case trước khi cải thiện là query “Tìm thông tư liên tịch 85/2016 quy định phối hợp giữa Bộ Tài chính và Bộ Công an trong bảo vệ an ninh quốc gia.” Nếu chỉ chạy vector search bằng `_mock_embed`, top-3 không chứa đúng Thông tư liên tịch 85/2016/TTLT-BTC-BCA. Sau khi thêm metadata filtering theo `Số hiệu` và header metadata trong chunk, query này trả về đúng văn bản ở top-1. Nếu làm tiếp, tôi sẽ lưu thêm metadata cấp chunk như `article_number`, `article_title` để filter sâu hơn đến từng điều.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 14 / 15 |
| **Tổng** | | **95 / 100** |

