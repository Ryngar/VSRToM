"""Prepare portable, metadata-cleaned Parquet files for the Hugging Face Hub.

No network requests or uploads. Dependencies: pyarrow and Pillow.
"""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

ALLOWED = {
    "sim": {"sample_id", "metadata", "question", "answer", "answer_options", "level", "task",
            "text-only", "text-pos", "text-dir", "vision_tool"},
    "real": {"question_id", "image_id", "image_url", "object_name", "object_ch_name", "order",
             "task_type", "question", "question_ch", "options", "answer", "persons", "observer", "observer_ch",
             "target_person", "target_person_ch"},
}
PRIVATE = re.compile(r"/(?:home|Users|scratch)/|[A-Za-z]:\\Users\\|\b(?:sk|hf|ghp)[_-][A-Za-z0-9_-]{16,}")
VALUE = {"dtype": "string", "_type": "Value"}
FEATURES = {key: dict(VALUE) for key in ["id", "scene_id", "question", "answer", "level", "task", "source_url", "annotation_json"]}
FEATURES["image"] = {"_type": "Image"}
FEATURES["options"] = [{key: dict(VALUE) for key in ["label", "text", "description", "use_case"]}]


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()
                if not (key.lower().endswith(("_path", "_dir")) or key.lower() in {"path", "error", "traceback"})}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, str) and PRIVATE.search(value):
        raise ValueError("Private path or credential remains in an annotation; value withheld")
    return value


def normalize(row, config, index):
    unknown = set(row) - ALLOWED[config]
    if unknown:
        raise ValueError("Unexpected source fields; review before exporting: " + ", ".join(sorted(unknown)))
    annotation = clean(row)
    if config == "sim":
        choices = [v for v in annotation["answer_options"].values() if isinstance(v, dict)]
        if len(choices) != 1:
            raise ValueError("Expected one Sim options category")
        options = choices[0]
        scene = str(annotation["sample_id"])
        level, task = annotation["level"], annotation["task"]
    else:
        options = annotation["options"]
        scene = str(annotation["image_id"])
        level, task = annotation["order"], annotation["task_type"]
    if level not in ("0-order", "1st-order", "2nd-order") or task not in ("visibility", "count", "layout"):
        raise ValueError("Invalid level/task")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", scene):
        raise ValueError("Unsafe scene identifier")
    source_url = annotation.pop("image_url", "")
    if source_url:
        url = urlsplit(source_url)
        if url.scheme not in ("http", "https") or url.hostname != "images.cocodataset.org" or url.username:
            raise ValueError("Unrecognized image source URL; review before release")
        source_url = urlunsplit((url.scheme, url.netloc, url.path, "", ""))
    answer = annotation["answer"].strip().upper()
    if answer not in options or not set(options) <= set("ABCD"):
        raise ValueError("Invalid answer/options")
    return {"id": f"{config}_{index:06d}", "scene_id": scene,
            "question": annotation["question"], "answer": answer, "level": level, "task": task,
            "options": [{"label": key, "text": value["text"], "description": value.get("description", ""),
                         "use_case": value.get("use_case")} for key, value in sorted(options.items())],
            "source_url": source_url,
            "annotation_json": json.dumps(annotation, ensure_ascii=False)}


@lru_cache(maxsize=8)
def portable_png(path):
    from PIL import Image
    raw = Path(path).read_bytes()
    with Image.open(BytesIO(raw)) as image:
        rgb = image.convert("RGB")
        pixels, size = rgb.tobytes(), rgb.size
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        # Keep the original pixel stream; discard ancillary metadata chunks.
        # Palette/transparency are preserved for the same RGB decoding.
        position = 8
        chunks = [raw[:8]]
        ended = False
        while position + 12 <= len(raw):
            length = int.from_bytes(raw[position:position + 4], "big")
            end = position + length + 12
            if end > len(raw):
                raise ValueError("Truncated PNG chunk")
            kind = raw[position + 4:position + 8]
            if kind in (b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS"):
                chunks.append(raw[position:end])
            elif not (kind[0] & 32):
                raise ValueError("Unsupported critical PNG chunk")
            position = end
            if kind == b"IEND":
                ended = True
                break
        if not ended:
            raise ValueError("PNG end marker missing")
        data = b"".join(chunks)
    else:
        # Real JPEGs are converted losslessly relative to decoded RGB pixels.
        fresh = Image.frombytes("RGB", size, pixels)
        buffer = BytesIO()
        fresh.save(buffer, format="PNG", compress_level=3)
        data = buffer.getvalue()
    with Image.open(BytesIO(data)) as restored:
        if restored.convert("RGB").tobytes() != pixels or restored.getexif():
            raise ValueError("Image pixel/metadata verification failed")
    return data


def schema():
    import pyarrow as pa
    fields = [pa.field(key, pa.string()) for key in FEATURES if key not in ("image", "options")]
    fields.extend([pa.field("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                   pa.field("options", pa.list_(pa.struct([(key, pa.string()) for key in ["label", "text", "description", "use_case"]])))])
    return pa.schema(fields, metadata={b"huggingface": json.dumps({"info": {"features": FEATURES}}).encode()})


def prepare(input_file, images, config, output=None, shard_mb=256, limit=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    extension = ".png" if config == "sim" else ".jpg"
    root = Path(images).resolve()
    writer = None
    shard_index, shard_bytes = 0, 0
    batch = []
    counts = Counter()
    seen = set()
    try:
        for index, line in enumerate(Path(input_file).open(encoding="utf-8")):
            if not line.strip():
                continue
            row = normalize(json.loads(line), config, sum(counts.values()))
            path = root / (row["scene_id"] + extension)
            if not path.is_file() or root not in path.resolve().parents:
                raise ValueError(f"Missing or invalid image for record {index + 1}")
            counts[f"{row['level']}-{row['task']}"] += 1
            seen.add(row["scene_id"])
            if output is not None:
                data = portable_png(str(path))
                row["image"] = {"bytes": data, "path": f"{config}_{row['scene_id']}.png"}
                if writer is None:
                    writer = pq.ParquetWriter(Path(output) / f"test-{shard_index:05d}.parquet", schema(), compression="zstd")
                batch.append(row)
                shard_bytes += len(data)
                if len(batch) >= 8:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema()), row_group_size=8)
                    batch.clear()
                    if shard_bytes >= shard_mb * 1024 * 1024:
                        writer.close(); writer = None
                        shard_index += 1; shard_bytes = 0
            if sum(counts.values()) % 100 == 0:
                print(config, "processed", sum(counts.values()), flush=True)
            if limit is not None and sum(counts.values()) >= limit:
                break
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema()), row_group_size=8)
    finally:
        if writer is not None:
            writer.close()
        portable_png.cache_clear()
    if not counts:
        raise ValueError("Empty input")
    return {"examples": sum(counts.values()), "unique_images": len(seen), "categories": dict(sorted(counts.items()))}


def card(repo_id, sample=False):
    return f'''---
language:
- en
configs:
- config_name: sim
  data_files:
  - split: test
    path: sim/test-*.parquet
- config_name: real
  data_files:
  - split: test
    path: real/test-*.parquet
tags:
- visual-spatial-reasoning
- theory-of-mind
- benchmark
---
# VSRToM

{'**SAMPLE EXPORT FOR VALIDATION ONLY.**' if sample else ''}

VSRToM evaluates 0-order egocentric, 1st-order allocentric and 2nd-order
recursive allocentric reasoning in synthetic and real-world scenarios.
Both `sim` and `real` configurations contain evaluation data in a `test` split.
No train/dev split is provided.

```python
from datasets import load_dataset
sim = load_dataset("{repo_id}", "sim", split="test")
real = load_dataset("{repo_id}", "real", split="test")
```

`image` contains embedded image bytes and a portable filename. `question`,
`options`, `answer`, `level` and `task` are standardized across configurations.
`annotation_json` contains the remaining research annotations in the original
schema, with crop paths and debugging fields removed. Parse it with `json.loads`.
This preserves orientation estimates, synthetic geometry and auxiliary text
representations. Ground-truth geometry and auxiliary text are reference
annotations and should not be supplied as inputs to image-only/default or
DePeR evaluation. For DePeR, use the tool-estimated orientations as specified
by the released task implementation.

`source_url` preserves the original COCO image URL for real-world images.
Original record and scene identifiers are for benchmark bookkeeping. Source
annotation key order is preserved so person-orientation hint order is reproducible.

## Preparation

Absolute local paths and crop filenames are removed. Images are exported as lossless PNG with embedded EXIF, XMP and text metadata
removed. Existing PNG pixel streams are retained; JPEGs are encoded from their
decoded RGB pixels. The exporter checks that decoded RGB pixels are unchanged.
These structural checks do not establish that image contents contain no
identifiable people or objectionable material.

## Source rights and publication status

This is a private preparation draft until the release owner completes review.
The code repository's Apache 2.0 license does not grant a license to source
images or third-party assets. Real image provenance and original image rights,
synthetic asset redistribution rights, and the license for new annotations must
be confirmed before a public release. No new blanket image license is asserted
by this preparation step.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for config in ("sim", "real"):
        parser.add_argument(f"--{config}-jsonl", type=Path, required=True)
        parser.add_argument(f"--{config}-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-id", default="YOUR_NAMESPACE/VSRToM")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--limit", type=int, help="Sample export for validation only")
    parser.add_argument("--shard-mb", type=int, default=256)
    args = parser.parse_args()
    if args.shard_mb < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Shard size and limit must be positive")
    if not args.check_only and args.output_dir is None:
        parser.error("--output-dir required unless --check-only")
    if not args.check_only:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {}
    for config in ("sim", "real"):
        output = None if args.check_only else args.output_dir / config
        if output is not None:
            output.mkdir()
        report[config] = prepare(getattr(args, f"{config}_jsonl"), getattr(args, f"{config}_images"),
                                 config, output, args.shard_mb, args.limit)
    if not args.check_only:
        (args.output_dir / "README.md").write_text(card(args.repo_id, sample=args.limit is not None), encoding="utf-8")
        manifest = {str(path.relative_to(args.output_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in args.output_dir.rglob("*.parquet")}
        (args.output_dir / "manifest.json").write_text(json.dumps({"statistics": report, "sha256": manifest}, indent=2) + "\n")
    print("VALIDATION_PASSED" if args.check_only else "EXPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
