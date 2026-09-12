from pathlib import Path
import pymupdf
from preprocessing import preprocess, copy_metadata_files

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract embedded text from one PDF."""
    pages: list[str] = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True).strip()

            pages.append(
                f"\n\n--- PAGE {page_number} ---\n\n{text}"
            )

    return "".join(pages).strip()

def extract_all_pdfs(input_directory: Path, output_directory: Path) -> None:
    """Extract text from every PDF under the input directory."""
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

    for pdf_path in pdf_paths:
        try:
            text = preprocess(extract_pdf_text(pdf_path))

            # Preserve the PDF's relative folder structure.
            relative_path = pdf_path.relative_to(input_directory)
            output_path = (
                output_directory / relative_path
            ).with_suffix(".txt")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")

            successful += 1
            print(f"[OK] {pdf_path} -> {output_path}")

        except Exception as error:
            failed += 1
            print(f"[FAILED] {pdf_path}: {error}")

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