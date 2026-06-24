"""Generate SPLADE vectors on a Colab GPU without connecting to PostgreSQL.

Upload this script and data/chunks to Google Drive, then run it in Colab.
The output JSONL can be imported locally with:
python scripts/06_index_chunks.py --skip-store --import-splade-jsonl <output.jsonl>
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        required=True,
        help="Directory containing *.children.jsonl files.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=768)
    return parser.parse_args()


def read_chunks(chunks_dir: Path) -> list[dict]:
    chunks = []
    for path in sorted(chunks_dir.glob("*/*/*.children.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                text = (chunk.get("retrieval_text") or "").strip()
                if text:
                    chunks.append({"chunk_id": chunk["chunk_id"], "text": text})
    return chunks


def completed_chunk_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed = set()
    with output_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                completed.add(json.loads(line)["chunk_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable. In Colab select a GPU runtime.")
    if args.batch_size < 1 or args.top_k < 1:
        raise ValueError("batch-size and top-k must be positive")

    chunks = read_chunks(args.chunks_dir)
    completed = completed_chunk_ids(args.output)
    pending = [chunk for chunk in chunks if chunk["chunk_id"] not in completed]
    print(
        f"Chunks: total={len(chunks):,} completed={len(completed):,} "
        f"pending={len(pending):,}"
    )
    if not pending:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).cuda().eval()
    dimension = int(model.config.vocab_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as output:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            encoded = tokenizer(
                [chunk["text"] for chunk in batch],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to("cuda")

            with torch.inference_mode():
                logits = model(**encoded).logits
                activations = torch.log1p(torch.relu(logits))
                weights = (
                    activations * encoded["attention_mask"].unsqueeze(-1)
                ).amax(dim=1)

            for chunk, row in zip(batch, weights):
                nonzero = torch.nonzero(row > 0, as_tuple=False).squeeze(-1)
                if nonzero.numel() > args.top_k:
                    _, positions = torch.topk(row[nonzero], args.top_k)
                    nonzero = nonzero[positions]
                nonzero, _ = torch.sort(nonzero)
                record = {
                    "chunk_id": chunk["chunk_id"],
                    "indices": nonzero.cpu().tolist(),
                    "values": row[nonzero].float().cpu().tolist(),
                    "dimension": dimension,
                    "model": args.model,
                }
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
            output.flush()
            done = min(start + len(batch), len(pending))
            print(f"Generated this run: {done:,}/{len(pending):,}")


if __name__ == "__main__":
    main()
