from src.chunking import ChunkingStrategyComparator

files = [
    "new_data/001_luat_52656.md",
    "new_data/002_nghi_dinh_8908.md",
    "new_data/003_nghi_dinh_14849.md",
]

comparator = ChunkingStrategyComparator()

for file_path in files:
    print("=" * 80)
    print(f"Tài liệu: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    result = comparator.compare(text, chunk_size=800)

    for strategy, stats in result.items():
        print(f"\nStrategy: {strategy}")
        print(f"Chunk Count: {stats['count']}")
        print(f"Avg Length: {stats['avg_length']:.2f}")