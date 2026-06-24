import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from finalrag.normalization.normalize import normalize_all


def main() -> None:
    results = normalize_all()
    print(f"\nNormalized sources: {len(results)}")
    for result in results:
        decisions = result.get("image_decisions") or {}
        image_summary = (
            f" | images kept={decisions.get('keep', 0)} "
            f"rejected={decisions.get('reject', 0)}"
            if decisions
            else ""
        )
        print(
            f"{result['domain']:<8} {result['source_type']:<4} "
            f"{result['source_name']}: elements={result['element_count']}"
            f"{image_summary}"
        )


if __name__ == "__main__":
    main()
