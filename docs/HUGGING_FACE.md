# Load, prepare and upload the dataset

The public benchmark is [Ryngar/VSRToM](https://huggingface.co/datasets/Ryngar/VSRToM).
Install the data dependencies and load either configuration:

```python
from datasets import load_dataset
sim = load_dataset("Ryngar/VSRToM", "sim", split="test")
real = load_dataset("Ryngar/VSRToM", "real", split="test")
```

VSRToM annotations use CC BY 4.0. Images and third-party assets are subject to
their original terms; consult the dataset card for details.

## Prepare a new export

The exporter writes a separate dataset folder; it does not upload or modify the
original data. Install the optional dependencies with `python -m pip install -e '.[data]'`.

```bash
python scripts/prepare_hf_dataset.py --sim-jsonl data/sim.jsonl --sim-images data/sim_images --real-jsonl data/real.jsonl --real-images data/real_images --repo-id YOUR_NAMESPACE/VSRToM --output-dir outputs/hf-dataset
```

Use `--check-only` to validate annotations/image references without writing data.
Use `--limit 3` with a new output directory for a small, clearly marked sample
export. Existing output directories are rejected to avoid mixing old and new
shards. A failed export must not be uploaded; a successful export prints
`EXPORT_COMPLETE` and writes a checksum manifest.

Image and option decoding was verified with datasets 5.1.0 and PyArrow 25.0.0.
Use the optional data dependencies rather than pairing a legacy datasets 2.x
installation with a recent PyArrow version.

The output contains a dataset card and two configurations, `sim` and `real`, each
with a `test` split. Images are embedded in Parquet shards. For portability,
Real `order`/`task_type` become `level`/`task`, and option records are normalized
to a list of label/text/description/use_case objects. Original research fields
remain accessible through `json.loads(row['annotation_json'])`.

Private crop paths and debugging fields are dropped; unexpected source fields
are rejected for review. Images are exported as lossless PNGs, removing embedded metadata while
verifying decoded RGB pixels. PNG pixel streams are retained and JPEGs are
encoded from their decoded RGB pixels. This can change
image encoding, file sizes and checksums relative to the original PNG/JPEG.
Image contents themselves still require human review before public release.

Upload from a machine that can connect to Hugging Face. Install the Hub client
and log in interactively; do not put a token into the script or Git repository:

```bash
python -m pip install -U huggingface_hub
hf auth login
```

Create a **private dataset repository** in the Hugging Face web UI. Select the
dataset repository type, confirm its visibility, and then upload only the
prepared output folder:

```bash
hf upload YOUR_NAMESPACE/VSRToM outputs/hf-dataset . --repo-type dataset
```

The uploader does not change an existing repository's visibility; verify it is
private before uploading. If the compute server has no outbound connectivity,
copy the prepared folder to a connected machine and run the same upload there.
The draft dataset card deliberately does not assert an unverified license for
source images. Confirm source-image and synthetic-asset rights and select a
license for new annotations before publishing the dataset publicly.

Loading a private dataset after authentication:

```python
from datasets import load_dataset
sim = load_dataset('YOUR_NAMESPACE/VSRToM', 'sim', split='test', token=True)
real = load_dataset('YOUR_NAMESPACE/VSRToM', 'real', split='test', token=True)
```

The original standalone evaluator consumes local JSONL/images; a Hub dataset
row can be converted back using `annotation_json`, with its embedded image saved
locally under the image convention described in `DATASET.md`. The annotations
include auxiliary references that must not be used as model input by accident.

## Export Hub rows for the standalone evaluator

The evaluator reads the original annotation schema and local images. To export
one Hub configuration while preserving decoded RGB pixels:

```python
import json
from pathlib import Path
from datasets import load_dataset

config = "sim"  # or "real"
rows = load_dataset("YOUR_NAMESPACE/VSRToM", config, split="test", token=True,
                    streaming=True)
output = Path("outputs/eval")
images = output / "images"
images.mkdir(parents=True, exist_ok=True)
with (output / "benchmark.jsonl").open("w", encoding="utf-8") as stream:
    for row in rows:
        doc = json.loads(row["annotation_json"])
        doc["image"] = row["id"] + ".png"
        row["image"].convert("RGB").save(images / doc["image"], format="PNG")
        stream.write(json.dumps(doc, ensure_ascii=False) + "\n")
```

Pass the resulting JSONL and image directory to `vsrtom.evaluate`; use task
`vsrtom` for `sim` and `vsrtom_real` for `real`. The relative `image` override
avoids relying on the original dataset's PNG/JPEG filename convention.

Source annotation key order is retained. Sorting the person-description keys
can change the orientation-hint order and therefore the actual model prompt.
