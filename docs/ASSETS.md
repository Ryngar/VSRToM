# External assets and tools

No external assets, models or checkpoints are redistributed by this code release.

## Blender resources

The original renderer expects this layout relative to `--assets-dir`:

```text
assets/base_scene_centered.blend
Sophia.blend
Man.blend
Basketball.blend
Obstacle.blend
```

These are resource names used by the renderer, not machine-specific paths.
The scene contains the original camera and lighting. Character resources expose
objects named `Sophia` and `Man`; the basketball and obstacle resources expose
collections/objects named `Basketball` and `Obstacle`. Appending uses the
collection named `Collection`, as in the research implementation.

The `.blend` files are not bundled because their redistribution terms have not
been verified for this release. You must supply compatible, authorized resources.
Changing meshes, camera settings or object scale changes the rendered distribution;
metadata and QA generation can run without any Blender assets.

## Direction estimation

The optional preprocessing script uses these external implementations:

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), via Transformers,
  for detecting the synthetic man and woman.
- [Orient Anything](https://github.com/SpatialVision/Orient-Anything), for estimating
  azimuth from each detected crop.

Install the upstream requirements in a dedicated environment. The wrapper
expects `vision_tower.DINOv2_MLP` and `inference.get_3angle`, with the large DINO
configuration and a compatible checkpoint. Pass the checkout and checkpoint
using CLI flags. The default image preprocessor is `facebook/dinov2-large`;
use `--dino-processor` if your upstream checkpoint specifies another processor.
Neither the external checkout nor the checkpoint should be committed here.

Detector box/text thresholds are 0.4/0.3, matching the source script. If multiple
detections have the same label, the last detection wins, matching the source;
images shared by multiple questions are processed once. Real-world person
annotation/preprocessing is not supplied by this synthetic helper: the evaluator
reads the existing `persons` field of the locally supplied Real data.

Review upstream code/model licenses and obtain resources from their official
sources. A license for this repository's wrapper does not grant permission to
redistribute upstream weights, source datasets, images or 3D assets.
