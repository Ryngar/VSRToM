"""Evaluate local benchmark JSONL with default or DePeR (no training)."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from .methods import predict
from .scoring import parse_vsrtom_response
from .tasks import build_prompt, image_path


def read_records(path):
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON at line {line_number}") from None
            if not isinstance(doc, dict):
                raise ValueError(f"Expected an object at line {line_number}")
            yield doc


def aggregate(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["category"]].append(record["correct"])
    categories = {key: {"n": len(values), "accuracy": sum(values) / len(values)}
                  for key, values in sorted(groups.items())}
    total = sum(v["n"] for v in categories.values())
    return {"n": total, "accuracy": sum(sum(v) for v in groups.values()) / total if total else None,
            "macro_accuracy": sum(v["accuracy"] for v in categories.values()) / len(categories) if categories else None,
            "categories": categories}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--task", choices=["vsrtom", "vsrtom_real"], default="vsrtom")
    parser.add_argument("--method", choices=["default", "deper"], default="default")
    parser.add_argument("--model", help="Hugging Face model ID or locally configured weights")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=1605632)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--dry-run", action="store_true", help="Validate records/images/prompts without loading weights")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_new_tokens <= 0 or args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        parser.error("Invalid token/pixel limits")
    if not args.dry_run and not args.model:
        parser.error("--model is required unless --dry-run is used")
    docs = []
    for doc in read_records(args.data):
        # Validate all selected inputs before creating outputs or loading weights.
        try:
            build_prompt(doc, args.task, args.method)
            image_path(doc, args.images, args.task)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(f"Record {len(docs) + 1}: {exc}")
        docs.append(doc)
        if args.limit and len(docs) >= args.limit:
            break
    if not docs:
        parser.error("No benchmark records")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / ("dry_run.json" if args.dry_run else "predictions.jsonl")
    if target.exists() or (not args.dry_run and (args.output_dir / "metrics.json").exists()):
        parser.error("Output already exists; choose a new output directory")
    # Do not serialize CLI paths, model cache paths, environment or original docs.
    config = {"task": args.task, "method": args.method, "n": len(docs),
              "generation": {"max_new_tokens": args.max_new_tokens, "do_sample": False,
                             "num_beams": 1, "min_pixels": args.min_pixels,
                             "max_pixels": args.max_pixels, "dtype": args.dtype,
                             "attn_implementation": args.attn_implementation}}
    if args.dry_run:
        target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"Validated {len(docs)} records and image references. No model loaded.")
        return
    from .models.qwen2_5_vl import Qwen25VL
    backend = Qwen25VL(args.model, device_map=args.device_map, dtype=args.dtype,
                       max_new_tokens=args.max_new_tokens, min_pixels=args.min_pixels,
                       max_pixels=args.max_pixels, attn_implementation=args.attn_implementation)
    scores = []
    with target.open("x", encoding="utf-8") as stream:
        for index, doc in enumerate(docs):
            prediction = predict(backend, doc, image_path(doc, args.images, args.task), args.task, args.method)
            extracted = parse_vsrtom_response(prediction.response)
            record = {"index": index, "category": f"{doc.get('level', doc.get('order'))}-{doc.get('task', doc.get('task_type'))}",
                      "answer": doc["answer"].strip().upper(), "prediction": extracted,
                      "correct": extracted == doc["answer"].strip().upper(), **asdict(prediction)}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            scores.append({"category": record["category"], "correct": record["correct"]})
            print(f"Completed {index + 1}/{len(docs)}", flush=True)
    (args.output_dir / "metrics.json").write_text(json.dumps({**config, **aggregate(scores)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
