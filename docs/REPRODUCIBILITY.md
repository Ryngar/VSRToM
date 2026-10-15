# Reproducibility and scope

This release is a refactoring of the research implementation into an independent
package. It does not contain a full fork of lmms-eval or the original experiment
workspace. It is not an archive of the exact execution environments for all paper
results.

## Preserved behavior

- Dataset geometry, QA templates and label functions, exclusion rules,
  answer/pattern sampling and option-content remapping come from the research code.
- The `default` and `deper` pre/post prompts for both tasks are retained verbatim
  in `vsrtom/prompts/`. The second-stage prompt and camera/person hints are also
  extracted from the source, including its original wording and spelling.
- Sim default has no orientation hint. Sim DePeR requires tool estimates.
  The original Real task appends available person orientations for both methods.
- DePeR's second call is gated by an ordered, case-sensitive map tag pair, not by
  an external ground-truth reasoning-level label. A response with no map uses
  round 1 directly. The original map slice includes `<cogmap>` but excludes
  `</cogmap>`; this formatting quirk is retained to preserve second-stage input.
- Round 2 is a new text-only request using the same Qwen model. It receives no
  image or full first-round reasoning/answer, just the extracted map and prompt.
- Answer extraction preserves the original parser: last complete answer tag,
  otherwise incomplete answer tag, otherwise boxed answer or free-text fallback.
  Results are exact option-letter accuracy, with unparseable answers incorrect.

## Release changes

- A single-example standalone Qwen backend replaces harness batching,
  distributed execution, custom CLI modifications, caching and debug hooks.
  Only still-image Qwen2.5-VL evaluation is included; no training, video, API
  provider clients, other model families, CoT baseline or ablation methods.
- All input/output locations and asset/checkpoint roots are CLI parameters.
  No environment dump, local model/cache paths, source image paths or source
  record metadata are copied into evaluation result records.
- Data validation rejects invalid records, missing directions for Sim DePeR,
  absolute image locations and images escaping the configured root. Unlike the
  original tool helper, missing orientations are not replaced with a generic
  error string containing a private path.
- Dataset script command-line entry points are parameterized. Generation's
  preset no longer overrides explicit sample counts unless requested.
- New evaluation outputs store both rounds and aggregate scores explicitly.
  The default budget is 1,024 tokens per call, as in the task YAML generation
  configuration. The larger fallback inside the original model class is not
  used as the standalone default. Pixel limits match that class's defaults:
  200,704 minimum, 1,605,632 maximum. Historical runs may have overridden these;
  inspect the relevant run configuration when reproducing a specific number.
- Blender uses a selectable CPU/CUDA mode and explicit collection-append arguments. The orientation wrapper retains the
  source model configuration, thresholds, crop rounding and per-label detection
  selection, while accepting external code/weights explicitly and reusing
  estimates for repeated image references.

## What is validated

The dependency-free tests exercise deterministic scene generation, geometric
visibility cases, exclusion/answer balance, export and option-label consistency,
DePeR's image-to-text-only transition, no-map fallback, task direction protocols,
answer selection and aggregation. CLI smoke checks validate both record formats.
They do not run Qwen, the detector, orientation model or Blender.

Full GPU inference and rendering require external dependencies, authorized
assets/data and weights. They have not been verified as part of this code-only
release. Reproducing paper numbers additionally requires the exact benchmark
selection, model revision, prompt/pixel/token configuration and runtime environment.
A regenerated candidate pool or the artificial smoke fixtures are not substitutes.
