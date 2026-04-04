import csv
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
SOURCE_DIR = RAW_DATA_DIR / "aclImdb"
OUTPUT_PATH = RAW_DATA_DIR / "imdb_reviews.csv"


def build_imdb_csv() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            "Expected raw IMDb directory at data/raw/aclImdb. "
            "Download and extract the Stanford aclImdb dataset first."
        )

    rows_written = 0
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=["review", "sentiment", "split", "source_file"],
        )
        writer.writeheader()

        for split in ["train", "test"]:
            for sentiment_folder in ["pos", "neg"]:
                folder = SOURCE_DIR / split / sentiment_folder
                for review_path in sorted(folder.glob("*.txt")):
                    writer.writerow(
                        {
                            "review": review_path.read_text(encoding="utf-8", errors="ignore").strip(),
                            "sentiment": "positive" if sentiment_folder == "pos" else "negative",
                            "split": split,
                            "source_file": review_path.name,
                        }
                    )
                    rows_written += 1

    print(f"Wrote {rows_written} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_imdb_csv()
