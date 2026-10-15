# Dataset construction and record format

## Synthetic pipeline

1. `metadata_generation.py` samples distinct scene configurations with two
   people, zero to three basketballs, and zero or one central obstacle.
   Positions and orientations retain the research implementation's grid.
   A fixed seed and configuration hash support reproducible generation.
2. `qa_generation.py` produces visibility, count and layout questions at each
   reasoning level. Its rules distinguish visible, behind, obstacle-blocked,
   ambiguous and overlapping positions. Default ambiguity tolerance is 1e-6;
   the obstacle-angle threshold is 135 degrees. Person blocking contributes
   to ambiguity. These are geometric labeling rules, not learned predictions.
3. Second-order answers require the man to see the woman. A ball contributes
   only when it is visible to both people. This implements
   `Vis(A,B) && Vis(A,C) && Vis(B,C)`. Ambiguous and no-sight cases are marked
   excluded. Layout is evaluated in the woman's frame for second-order items.
4. `filtering_and_sampling.py` discards an entire scene if any of its nine
   answers is excluded. It forms a pattern from the nine answer labels and
   four counts for balls hidden behind people or by obstacles. Within each
   level/task category, it allocates nearly equal quotas across available
   answer classes and then balances patterns using layered allocation.
   Quotas are not automatically transferred from exhausted answer groups.
5. `shuffle.py` moves the correct answer content to a least-used answer label
   within each task group and randomizes the other option positions. This is
   performed on the combined benchmark file; scoring uses the updated label.
6. `render.py` renders selected scene IDs with Blender. Scene metadata are
   labels, and are never inserted into the default or DePeR prompts. The
   optional orientation stage estimates directions from rendered images.

No training or development split is constructed by these scripts. The
result is a benchmark evaluation collection. Seeds, selected scene IDs and
allocation statistics are saved locally by the construction pipeline.

Use `--preset test`, `small`, `medium`, or `large` for predefined candidate
counts. Without `--preset`, explicit count arguments are honored. The release
changes the original automatic large-preset default so a custom count is not
silently overwritten. All count arguments retain their original names; see
`python -m vsrtom.dataset.metadata_generation --help`.

## Synthetic evaluation records

Each line is a JSON object containing:

- `sample_id`: scene identifier, used only for local image lookup.
- `question`: natural-language question.
- `answer`: correct option letter, A-D.
- `answer_options`: exactly one task entry, such as
  `{"count": {"A": {"text": "0", "description": "No visible basketballs"}, ...}}`.
- `level`: `0-order`, `1st-order`, or `2nd-order`.
- `task`: `visibility`, `count`, or `layout`.
- `image`: optional relative image filename, overriding `<sample_id>.png`.
- `vision_tool`: required for Sim DePeR, with `man_orientation.azimuth` and
  `woman_orientation.azimuth` as tool-estimated angles in [0, 360]. Sim default
  does not use this field. Null estimates trigger a validation error for DePeR.

Generation also exports `metadata` for downstream processing. The evaluation
prompt builder does not consume it. Original questions may share an image;
results are indexed by input row, not by scene ID alone.

## Real evaluation records

The question and answer follow the same convention. Real source files may use
`order` and `task_type`; the evaluator accepts these as aliases for `level`
and `task`. The Hugging Face exporter standardizes them.
`options` directly maps letters to objects with `text` and `description`
(and optional `use_case`). Use `image_id` for `<image_id>.jpg`, or set a relative
`image` filename. Available person directions have this shape:

```json
{
  "persons": {
    "a person wearing a blue shirt": {"orientation": {"azimuth": 90}}
  }
}
```

The source task excludes generic `person_*` entries when constructing the
orientation text. Real directions are rounded to eight standard azimuths,
while Sim directions are rounded to four. Both use 0 degrees = south,
90 = west, 180 = north, and 270 = east. The original Real formatter's
behavior of appending available directions for both methods is retained.

Only portable image names are accepted. Absolute image paths, traversal outside
the configured image root and remote image URLs are rejected. Existing private
benchmark dumps should be converted locally to this schema without publishing
their original filesystem metadata.
