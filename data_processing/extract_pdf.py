from pathlib import Path
import pymupdf
from data_processing.preprocessing import (
    build_vocabulary,
    copy_metadata_files,
    preprocess,
    repair_ligatures,
)

# Wider than a paragraph indent, narrower than a column gutter.
COLUMN_GAP = 40


def page_text_by_columns(page, gap: int = COLUMN_GAP) -> str:
    """Read each column top-to-bottom. Sorting by position alone interleaves them."""
    blocks = [b for b in page.get_text("blocks") if b[6] == 0]

    if not blocks:
        return page.get_text("text", sort=True)

    # A wide jump between consecutive block left edges starts a new column.
    lefts = sorted(b[0] for b in blocks)
    splits = [lefts[i + 1] for i in range(len(lefts) - 1) if lefts[i + 1] - lefts[i] > gap]
    bounds = [0.0] + splits + [page.rect.width + 1]

    lines: list[str] = []
    for low, high in zip(bounds, bounds[1:]):
        column = [b for b in blocks if low <= b[0] < high]
        lines.extend(b[4] for b in sorted(column, key=lambda b: b[1]))

    return "\n".join(lines)


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract embedded text from one PDF."""
    pages: list[str] = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page_text_by_columns(page).strip()

            pages.append(
                f"\n\n--- PAGE {page_number} ---\n\n{text}"
            )

    return "".join(pages).strip()

def extract_all_pdfs(input_directory: Path, output_directory: Path, progress_callback=None) -> None:
    """Extract text from every PDF under the input directory.

    progress_callback(index, total, pdf_path, status), called after each file,
    lets a caller (e.g. a background ingest job) report progress; unused by
    the CLI entry point below.
    """
    if not input_directory.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_directory}"
        )

    output_directory.mkdir(parents=True, exist_ok=True)

    # rglob searches the directory and all nested subdirectories.
    pdf_paths = sorted(input_directory.rglob("*.pdf"))

    if not pdf_paths:
        print(f"No PDF files found under: {input_directory}")
        return

    print(f"Found {len(pdf_paths)} PDF files.")

    successful = 0
    failed = 0

    # Extracted first so the whole corpus can seed the vocabulary that validates
    # ligature repairs.
    extracted = {}
    for pdf_path in pdf_paths:
        try:
            extracted[pdf_path] = extract_pdf_text(pdf_path)
        except Exception as error:
            failed += 1
            print(f"[FAILED] {pdf_path}: {error}")

    vocabulary = build_vocabulary(extracted.values())
    print(f"Vocabulary: {len(vocabulary)} words.")

    for index, (pdf_path, raw_text) in enumerate(extracted.items(), start=1):
        try:
            text, repairs = repair_ligatures(raw_text, vocabulary)
            if repairs:
                print(f"[REPAIRED] {pdf_path.name}: {repairs} mis-encoded words")

            text = preprocess(text)

            # Preserve the PDF's relative folder structure.
            relative_path = pdf_path.relative_to(input_directory)
            output_path = (
                output_directory / relative_path
            ).with_suffix(".txt")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")

            successful += 1
            print(f"[OK] {pdf_path} -> {output_path}")
            status = "ok"

        except Exception as error:
            failed += 1
            print(f"[FAILED] {pdf_path}: {error}")
            status = "failed"

        if progress_callback:
            progress_callback(index, len(extracted), pdf_path, status)

    print("\nExtraction complete.")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    extract_all_pdfs(
        input_directory=Path("papers"),
        output_directory=Path("processed_text"),
    )
    copy_metadata_files(
        input_directory=Path("papers"),
        output_directory=Path("processed_text"),
    )