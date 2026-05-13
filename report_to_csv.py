import argparse
import csv
import re
from pathlib import Path


METRIC_COLUMNS = [
    "raw_zipf_s",
    "raw_zipf_r2",
    "raw_mand_s",
    "raw_mand_q",
    "raw_mand_r2",
    "lemma_zipf_s",
    "lemma_zipf_r2",
    "lemma_mand_s",
    "lemma_mand_q",
    "lemma_mand_r2",
]

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

LANG_RE = re.compile(r"^=== LANGUAGE: (?P<language>.+?) ===$")
ZIPF_RE = re.compile(
    rf"^(?P<label>RAW|LEMMA) Zipf:\s*"
    rf"s\s*[^0-9+\-.]*(?P<s>{NUMBER}),\s*"
    rf"R\^2\s*[^0-9+\-.]*(?P<r2>{NUMBER})"
)
MANDELBROT_RE = re.compile(
    rf"^(?P<label>RAW|LEMMA) Zipf-Mandelbrot:\s*"
    rf"s\s*[^0-9+\-.]*(?P<s>{NUMBER}),\s*"
    rf"q\s*[^0-9+\-.]*(?P<q>{NUMBER}),\s*"
    rf"C\s*[^0-9+\-.]*(?P<c>{NUMBER}),\s*"
    rf"R\^2\s*[^0-9+\-.]*(?P<r2>{NUMBER})"
)


def latest_report() -> Path:
    reports = sorted(Path(".").glob("results_*/report.txt"), reverse=True)
    if not reports:
        raise FileNotFoundError("No report.txt found in results_* directories.")
    return reports[0]


def empty_row(language: str) -> dict[str, str]:
    row = {"language": language}
    row.update({column: "" for column in METRIC_COLUMNS})
    return row


def parse_report(report_path: Path) -> list[dict[str, str]]:
    rows = []
    current_row = None

    with report_path.open("r", encoding="utf-8") as report:
        for line in report:
            line = line.strip()

            lang_match = LANG_RE.match(line)
            if lang_match:
                if current_row is not None:
                    rows.append(current_row)
                current_row = empty_row(lang_match.group("language"))
                continue

            if current_row is None:
                continue

            zipf_match = ZIPF_RE.match(line)
            if zipf_match:
                prefix = "raw" if zipf_match.group("label") == "RAW" else "lemma"
                current_row[f"{prefix}_zipf_s"] = zipf_match.group("s")
                current_row[f"{prefix}_zipf_r2"] = zipf_match.group("r2")
                continue

            mandelbrot_match = MANDELBROT_RE.match(line)
            if mandelbrot_match:
                prefix = "raw" if mandelbrot_match.group("label") == "RAW" else "lemma"
                current_row[f"{prefix}_mand_s"] = mandelbrot_match.group("s")
                current_row[f"{prefix}_mand_q"] = mandelbrot_match.group("q")
                current_row[f"{prefix}_mand_r2"] = mandelbrot_match.group("r2")

    if current_row is not None:
        rows.append(current_row)

    if not rows:
        raise ValueError(f"No language blocks found in {report_path}")

    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path, include_language: bool) -> None:
    columns = METRIC_COLUMNS if not include_language else ["language", *METRIC_COLUMNS]
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CSV summary from a Zipf experiment report.txt file."
    )
    parser.add_argument(
        "report",
        nargs="?",
        type=Path,
        help="Path to report.txt. If omitted, the newest results_*/report.txt is used.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to output CSV. Defaults to summary.csv next to report.txt.",
    )
    parser.add_argument(
        "--no-language",
        action="store_true",
        help="Write only metric columns, without the leading language column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.report or latest_report()
    output_path = args.output or report_path.with_name("summary.csv")

    rows = parse_report(report_path)
    write_csv(rows, output_path, include_language=not args.no_language)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
