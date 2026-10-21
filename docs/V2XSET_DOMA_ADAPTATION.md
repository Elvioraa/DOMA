# V2XSet DOMA V2 adaptation

## Scope and compatibility

This adaptation adds an isolated five-config DOMA V2 family for V2XSet:

- `stage1/m1.yaml`: train the m1 PointPillar anchor and shared DOMA V2 modules.
- `stage2/m2.yaml`: adapt the EfficientNet LSS camera member.
- `stage2/m3.yaml`: adapt the SECOND LiDAR member.
- `stage2/m4.yaml`: adapt the ResNet-101 LSS camera member.
- `final_infer/m1m2m3m4.yaml`: instantiate all four members for merged,
  ordered collaborative inference.

The implementation reuses `V2XSETBaseDataset`,
`DOMAHeterPyramidSingle`, `DOMAHeterPyramidCollab`, the DOMA V2 loss,
the existing fail-closed DOMA checkpoint merger, and the DOMA ordered
inference entry. No HEAL, OPV2V DOMA, dataset loader, trainer, or evaluator
file is changed by this adaptation.

Static compatibility is supported. Real V2XSet loading, CUDA/spconv model
construction, training, checkpoint merge, and AP evaluation remain server
validation items.

## Configuration derivation

Each new YAML starts from the matching V2XSet HEAL YAML. Only the experiment
name, model entry, DOMA method block, and the Stage1/Stage2 loss entry differ.
The read-only checker compares the remaining parsed configuration tree against
HEAL and fails if any data, sensor, training, anchor, NMS, communication, or
detection setting drifts.

Stage1 and all three Stage2 configs use
`doma_point_pillar_pyramid_loss`. The Final config keeps the V2XSet HEAL
`point_pillar_pyramid_loss`; the ordered inference entry does not construct
or call a training criterion, and retaining it avoids introducing the
OPV2V-only depth-loss contract into a no-depth V2XSet config.

| Modality | V2XSet sensor contract | DOMA role |
|---|---|---|
| m1 | LiDAR, PointPillar, 0.4 x 0.4 x 4 voxel | Stage1 identity anchor |
| m2 | Four cameras, EfficientNet LSS, 384 x 512 | Stage2 object/context adapters |
| m3 | LiDAR, SECOND, 0.1 m cubic voxel | Stage2 object/context adapters |
| m4 | Four cameras, ResNet-101 LSS, 336 x 448 | Stage2 object/context adapters |

V2XSet camera configs use `input_source: [camera]` and keep depth
supervision disabled because the released V2XSet layout does not provide the
OPV2V-H depth extension used by the OPV2V DOMA configs.

V2XSet m3 intentionally has no `heter.lidar_channels_dict`. Therefore
`Adaptor.switch_lidar_channels` returns the original `.pcd` path and never
rewrites it to `OPV2V_Hetero/*_32.pcd`. The m1/m3 distinction is model
heterogeneity (PointPillar versus SECOND and different voxelization), not a
claim of 64-line versus 32-line sensor heterogeneity. No V2XSet point-cloud
downsampling is required by these configs.

The V2XSet HEAL ranges and postprocessing are retained:

- Stage1: `[-102.4, -51.2, -3, 102.4, 51.2, 1]`.
- Stage2 m2/m4: `[-51.2, -51.2, -3, 51.2, 51.2, 1]`.
- Stage2 m3: `[-96, -48, -3, 96, 48, 1]`.
- Final: `[-102.4, -51.2, -3, 102.4, 51.2, 1]`.
- Anchor, target thresholds, `max_num: 150`, and `nms_thresh: 0.15`
  are unchanged from V2XSet HEAL.

## Data and modality assignments

All configs use:

```text
dataset/V2XSET/train
dataset/V2XSET/validate
dataset/V2XSET/test
```

and the base assignment path:

```text
opencood/logs/heter_modality_assign/v2xset_4modality.json
```

The existing generators support the Linux server layout. Generate the two
runtime artifacts from the repository root; do not add them to Git:

```bash
python - <<'PY'
from opencood.utils.heter_utils import (
    assign_modality_4,
    assign_modality_4_in_order,
)

assign_modality_4(
    "dataset/V2XSET",
    "opencood/logs/heter_modality_assign/v2xset_4modality.json",
)
assign_modality_4_in_order(
    "dataset/V2XSET",
    "opencood/logs/heter_modality_assign/v2xset_4modality_in_order.json",
)
PY
```

`assign_modality_4` covers train, validate, and test with the existing fixed
seed. `assign_modality_4_in_order` covers test only. It sorts CAV IDs, moves
the infrastructure CAV `-1` behind vehicles, and assigns m1, m2, m3, m4
circularly. Thus the first vehicle is m1 and remains eligible as ego.

The ordered inference scripts replace the base assignment suffix with
`_in_order.json` and evaluate `use_cav` values 1, 2, 3, and 4. The
inference dataset uses all in-range CAVs to form GT but only the first
`use_cav` members as model inputs.

When any camera modality is enabled, `V2XSETBaseDataset` removes scenario
`2021_09_09_13_20_58` because it has incomplete camera files. The V2XSet
HEAL and DOMA camera/final configs therefore use the same filtered scenarios.
LiDAR-only Stage1 and m3 do not trigger that filter and naturally have a
different sample count from camera-enabled stages.

## Checkpoint workflow

Use the existing HEAL staged layout. Stage1 is trained from its own config.
For each Stage2 modality, create a separate run directory, place the selected
same-run Stage1 checkpoint there as `net_epoch1.pth` (or link to a protected
copy), and save the matching Stage2 YAML as `config.yaml`. Starting each
Stage2 with its own directory prevents m2/m3/m4 runs from overwriting one
another. The ordinary trainer loads the Stage1 state with `strict=False`;
the DOMA model then freezes Stage1-owned shared modules and trains only the
active modality's object and context adapters together with the existing HEAL
Stage2 path.

After all four training outputs have been reviewed, create a fresh Final
directory containing the Final YAML as `config.yaml`, then merge in the
required order:

```bash
python -B -m opencood.tools.doma_tools merge_final \
  /abs/path/to/stage2/m2 \
  /abs/path/to/stage2/m3 \
  /abs/path/to/stage2/m4 \
  /abs/path/to/stage1/m1 \
  /abs/path/to/final_infer
```

The merger checks an identical normalized DOMA method fingerprint, exact
equality of frozen Stage1 shared tensors, and complete modality-owned
adapters. It writes `net_epoch1.pth` only after those checks pass.

## Static and data checks

Local static validation:

```bash
python -B -m opencood.tools.check_doma_v2xset
python -m py_compile opencood/tools/check_doma_v2xset.py
git diff --check
```

On the server, after the real data and generated assignment JSON files exist:

```bash
python -B -m opencood.tools.check_doma_v2xset \
  --check-data \
  --data-root dataset/V2XSET \
  --sample-scenarios 1
```

This scan is read-only. It reports split/scenario/sample counts and checks each
sampled CAV for a timestamp YAML, LiDAR PCD, four camera images, four camera
calibrations, poses, and vehicle labels. It also verifies assignment coverage,
legal modalities, test-only in-order coverage, infrastructure ordering, and
ego=m1.

A dependency-available model-construction smoke test may be run without
training. The camera encoder can populate the user's PyTorch pretrained-weight
cache when the weights are absent, so run this only after the server cache or
approved weight source is ready:

```bash
python - <<'PY'
from pathlib import Path
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools.train_utils import create_model

root = Path("opencood/hypes_yaml/v2xset/MoreModality/DOMA/V2")
for config in sorted(root.rglob("*.yaml")):
    model = create_model(load_yaml(str(config)))
    print(config, model.__class__.__name__)
PY
```

For a single-item dataset smoke test, use the final config, set
`validate_dir=test_dir`, set `use_cav=1`, and replace its assignment path
with the in-order JSON before calling `build_dataset(..., train=False)[0]`.
This exercises real file reading but is not an AP or full inference run.

## Ordered inference range caveat

`inference_heter_in_order.py` and
`inference_doma_heter_in_order.py` share the same legacy behavior: both
override the YAML range from the `--range` argument and set communication
range to 180 m. Their default `--range 204.8,102.4` is not the V2XSet YAML
range. To preserve the configured V2XSet evaluation extent, both HEAL and
DOMA comparisons must be invoked explicitly with:

```bash
--range 102.4,51.2 --use_cav "[1,2,3,4]"
```

No common inference code is changed here because the two methods already use
the same data selection, GT construction, postprocessor, and AP accumulator.
Changing the generic defaults would alter existing OPV2V behavior. A real
server comparison must record the explicit range, checkpoint, assignment JSON,
filtered sample count, and AP output for both methods.
