"""
Demo: Best-fit document packing (simplified version of dataloader lines 119-148).

Goal: Pack variable-length "documents" into fixed-length rows with NO PADDING.
- Each row has capacity row_capacity (e.g. 10 tokens).
- Algorithm per row:
  1. Pick the LARGEST document from the buffer that fits in remaining space -> use it whole.
  2. Repeat until no document fits.
  3. When nothing fits, CROP the SHORTEST document to fill the remaining space exactly.

This minimizes wasted space and avoids padding (100% utilization).
"""

def pack_one_row(doc_buffer, row_capacity):
    """Pack documents into a single row using best-fit. Modifies doc_buffer in place."""
    row = []
    pos = 0
    while pos < row_capacity:
        remaining = row_capacity - pos
        # Find largest doc that fits entirely
        best_idx = -1
        best_len = 0
        for i, doc in enumerate(doc_buffer):
            doc_len = len(doc)
            if doc_len <= remaining and doc_len > best_len:
                best_idx = i
                best_len = doc_len

        if best_idx >= 0:
            doc = doc_buffer.pop(best_idx)
            doc_len = len(doc)
            row.extend(doc)
            pos += doc_len
        else:
            # No doc fits -> crop shortest to fill remaining
            if not doc_buffer:
                break
            shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
            doc = doc_buffer.pop(shortest_idx)
            row.extend(doc[:remaining])
            pos += remaining
    return row


def main():
    # Example: row capacity 10, batch size B=2
    row_capacity = 10
    B = 2

    # Fake "documents" = lists of token IDs (we use 1,2,3... for readability)
    # In real code these come from tokenizer; lengths vary
    doc_buffer = [
        [1, 2],           # len 2
        [3, 4, 5],        # len 3
        [6, 7, 8, 9],     # len 4
        [10, 11],         # len 2
        [12, 13, 14],     # len 3
        [15, 16, 17, 18, 19],  # len 5
        [20, 21],         # len 2
        [22, 23, 24],     # len 3
    ]

    print("Best-fit packing demo")
    print("Row capacity:", row_capacity, "| Batch size B:", B)
    print("Initial buffer (doc lengths):", [len(d) for d in doc_buffer])
    print()

    for row_idx in range(B):
        row = pack_one_row(doc_buffer, row_capacity)
        print(f"Row {row_idx}: {row}  (len={len(row)})")
        print(f"  Buffer after: {len(doc_buffer)} docs, lengths = {[len(d) for d in doc_buffer]}")
    print("Done. Each row is exactly", row_capacity, "tokens (or less if buffer ran out).")


if __name__ == "__main__":
    main()
