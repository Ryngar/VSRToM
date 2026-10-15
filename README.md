# VSRToM

Dataset construction and default / **DePeR** evaluation for VSRToM, covering
0-order egocentric, 1st-order allocentric and 2nd-order recursive allocentric
reasoning in synthetic and real-world scenarios.

This is a standalone code release extracted from the research implementation.
It includes the synthetic scene metadata generator, rule-based QA labeling,
filtering and balanced sampling, option randomization, Blender rendering entry
point, task prompts/scoring, and the Qwen2.5-VL implementation of default and
DePeR. It does not require a modified lmms-eval installation.

No benchmark data, real photographs, pretrained weights, 3D assets, original
experiment logs or participant records are bundled. The code has no embedded
credentials, machine-specific paths or author contact information.

## Dataset

The benchmark is available at [Ryngar/VSRToM on Hugging Face](https://huggingface.co/datasets/Ryngar/VSRToM),
with `sim` and `real` configurations. New VSRToM annotations are licensed under
CC BY 4.0; images and third-party assets retain their respective original terms.
See the dataset card for component-specific licensing and
[loading instructions](docs/HUGGING_FACE.md).

## Installation

Use Python 3.10 or later, from the repository root:

```bash
python -m pip install -e .
```

Dataset generation and the unit tests use only the Python standard library.
For Qwen evaluation, install a PyTorch build compatible with your hardware and:

```bash
python -m pip install -e '.[qwen]'
```

The optional extra selects Transformers 4.51.3 and qwen-vl-utils 0.0.14. It is a
release dependency specification, not a recovered lockfile for every paper run.
See [reproducibility notes](docs/REPRODUCIBILITY.md). The Qwen adapter follows the
[Transformers Qwen2.5-VL interface](https://huggingface.co/docs/transformers/v4.51.3/model_doc/qwen2_5_vl).
Blender and the orientation tools are separate optional dependencies.

## Construct synthetic questions

All commands below run from the repository root and write local artifacts into
`outputs/`, which is ignored by Git.

```bash
mkdir -p outputs/build
python -m vsrtom.dataset.metadata_generation --preset medium --output_file outputs/build/metadata.jsonl
python -m vsrtom.dataset.qa_generation --input outputs/build/metadata.jsonl --output outputs/build/qa_data.jsonl
python -m vsrtom.dataset.filtering_and_sampling --input outputs/build/qa_data.jsonl --output-dir outputs/build/balanced --per-category 8 --seed 42
python -m vsrtom.dataset.shuffle --input outputs/build/balanced/VSRToM_Bench.jsonl --output outputs/build/benchmark.jsonl --seed 42
python -m vsrtom.dataset.statistic --input outputs/build/benchmark.jsonl
```

The example uses small quotas to check the pipeline. Increase the candidate
pool and per-category quota for a larger dataset. Sampling reports actual
counts and warns if a quota cannot be met; inspect its statistics rather than
assuming all requested samples were obtained. This regeneration example does
not recreate the exact selected benchmark identities from the paper.

The filtering step removes scenes with any excluded or ambiguous answer,
balances answer classes within each of the nine categories, and diversifies
joint answer/visibility patterns. Option randomization updates the correct
answer along with its content. See [dataset construction](docs/DATASET.md).

## Render images and estimate directions

The Blender renderer retains the scene placement from the research code.
Provide the required assets separately as described in [assets](docs/ASSETS.md):

```bash
blender --background --python vsrtom/dataset/render.py -- --metadata outputs/build/metadata.jsonl --selected-dir outputs/build/balanced --assets-dir assets --output-dir outputs/images --device GPU
```

Use `--device CPU` if CUDA rendering is unavailable. Rendering requires the
specified `.blend` resources; this repository does not include them.

Synthetic DePeR requires **tool-estimated** person orientations. With an
externally installed Orient Anything checkout and its checkpoint:

```bash
python -m vsrtom.orientation --input outputs/build/benchmark.jsonl --output outputs/build/benchmark_with_orientations.jsonl --images outputs/images --orient-repo external/Orient-Anything --orient-checkpoint weights/orient-anything.pt
```

Follow the upstream installation instructions and licenses linked in
[assets](docs/ASSETS.md), preferably using a separate environment. Missing
detections remain null and must be resolved before DePeR evaluation. The
pipeline does not silently replace missing predictions with ground-truth
scene directions.

## Evaluate default and DePeR

Supply benchmark JSONL and images locally. Synthetic records use
`answer_options`, `sample_id`, and images named `<sample_id>.png`. Real records
use `options`, `image_id`, and `<image_id>.jpg`. A relative `image` field may
override either filename convention. See [record schemas](docs/DATASET.md).

Validate inputs without loading model weights:

```bash
python -m vsrtom.evaluate --data data/sim.jsonl --images data/images --task vsrtom --method default --output-dir outputs/check-default --dry-run
python -m vsrtom.evaluate --data data/sim_with_orientations.jsonl --images data/images --task vsrtom --method deper --output-dir outputs/check-deper --dry-run
```

Run Qwen2.5-VL:

```bash
python -m vsrtom.evaluate --model Qwen/Qwen2.5-VL-32B-Instruct --data data/sim.jsonl --images data/images --task vsrtom --method default --output-dir outputs/qwen-default
python -m vsrtom.evaluate --model Qwen/Qwen2.5-VL-32B-Instruct --data data/sim_with_orientations.jsonl --images data/images --task vsrtom --method deper --output-dir outputs/qwen-deper
```

For real-world data, set `--task vsrtom_real`. The original Real task appends
available `persons` directions for **both default and DePeR**; the Sim default
prompt has no direction hints. These details are preserved in this release.
The Real DePeR prompt retains the original generic man/woman hint wording;
actual per-person directions are appended from `persons` when available.

Default makes one image-conditioned call. DePeR first makes an image-conditioned
call. If that response contains an ordered `<cogmap>...</cogmap>` pair, it makes
a fresh **text-only** second call using that map, the original question/options,
and the original reasoning hints. Otherwise it uses the first response. Round 1
can contain reasoning and a preliminary answer as well as the cognitive map.
The final answer parser uses the last answer tag, so the second round takes
precedence when present.

`predictions.jsonl` records the two rounds separately; `metrics.json` reports
per-category accuracy, overall example-weighted accuracy, and category-macro
accuracy. Decoding is greedy, batch size is one, and the default generation
budget is 1,024 new tokens **per call**. Token and pixel limits can be changed
explicitly; record any changes when comparing runs. New runs require a fresh
output directory.

## Checks without model downloads

```bash
python -m unittest discover -s tests -v
python examples/make_smoke_data.py --output-dir outputs/smoke
python -m vsrtom.evaluate --data outputs/smoke/sim.jsonl --images outputs/smoke --method default --output-dir outputs/smoke-default --dry-run
python -m vsrtom.evaluate --data outputs/smoke/sim.jsonl --images outputs/smoke --method deper --output-dir outputs/smoke-deper --dry-run
python -m vsrtom.evaluate --data outputs/smoke/real.jsonl --images outputs/smoke --task vsrtom_real --method deper --output-dir outputs/smoke-real --dry-run
```

The smoke fixtures are artificial empty images with invented fields. They
exercise I/O and prompt construction only and must not be used as a benchmark.

## Public release

The repository uses Apache License 2.0 for code and retains the lmms-eval
attribution in [NOTICE](NOTICE) and [third-party license](third_party/lmms-eval-LICENSE).
This does not license external data, assets or model weights. Their publication
and redistribution need to be handled separately.

Before publishing a staged Git snapshot:

```bash
python scripts/audit_release.py --git-index
```

See [release scope](docs/RELEASE.md) for exclusions and the audit's limits.

For image-and-question dataset packaging, see [Hugging Face preparation](docs/HUGGING_FACE.md).
