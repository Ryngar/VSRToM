# Public release scope

Included:

- Synthetic scene/QA generation, filtering, balanced sampling and option shuffle.
- Configurable Blender renderer and optional synthetic orientation preprocessing.
- Sim/Real task formatters, retained default/DePeR prompts and response scoring.
- Standalone Qwen2.5-VL backend and default/DePeR two-stage execution.
- Installation/usage notes, license attribution, tests and a static release scan.

Excluded:

- Original Git history, remote configuration and commit author identities.
- Credentials, API clients, server configuration, absolute private paths and
  private model/cache directory names.
- Raw benchmark JSONL, source/real images, participant and annotation records,
  original IDs/logs/results, model weights, Blender files and downloaded assets.
- Other baselines, intermediate experiments, ablations and unrelated model code.

The public repository starts with a code-only snapshot and excludes the original research history.
The code license is Apache 2.0; third-party attribution is retained separately.
Data and third-party resources are outside this license grant and release scope.

## Checking a publication snapshot

Review `git status` and the exact staged file list, then run:

```bash
python scripts/audit_release.py --git-index
```

The scanner reports filenames, line numbers and finding types without displaying
matching secrets. It checks common private paths, emails, server aliases, IP
addresses, token patterns, key blocks, unexpected artifacts and oversized files.
It does not establish that arbitrary text, images or datasets contain no personal
information. The initial code-only snapshot is also manually reviewed for scope.

`.gitignore` excludes common datasets, weights, images, results, keys and local
configuration. Git ignore rules do not remove files already tracked or added
with force. Inspect later changes before publishing; do not copy private dumps
or raw experiment folders into the release.

The tests and example generator create their own artificial data at runtime.
Evaluation logs can contain generated descriptions and reasoning. Keep those
local unless they have been separately reviewed for publication.
