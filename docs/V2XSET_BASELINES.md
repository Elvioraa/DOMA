# V2XSet independent-detector No Fusion / Late Fusion

These are four independently trained `heter_model_late` detectors, not HEAL/DOMA
Pyramid checkpoint conversions. No fifth detector is trained. Run commands from
`/home/fkk/my_data/disk2/code/DOMA` in the existing training Python environment.
No command below creates screen/tmux or starts a background process.

## Configuration and protocol

Source directory: `opencood/hypes_yaml/v2xset/Baselines/`.

| YAML | Detector | Batch | Adam LR | Output under `opencood/logs/V2XSET_BASELINES_SEED303/` |
|---|---|---|---|---|
| `m1_pointpillar.yaml` | PointPillar | 16 | 0.002 | `single/m1_pointpillar/` |
| `m2_lss_effnet.yaml` | LSS EfficientNet | 4 | 0.002 | `single/m2_lss_effnet/` |
| `m3_second.yaml` | SECOND | 16 | 0.002 | `single/m3_second/` |
| `m4_lss_resnet101.yaml` | LSS ResNet101 | 4 | 0.002 | `single/m4_lss_resnet101/` |
| `final.yaml` | all four branches | evaluation only | — | `final_infer/` |

The check tool verifies optimizer, LR schedule, batch size and model settings
against the original Single YAMLs. All independent training configurations use
seed 303, 30 epochs, eval/save frequency 2, max_cav 5 and comm_range 180.
Native training ranges/voxel settings and learning-rate milestones are retained.
Automatic inference after training is disabled. The existing trainer evaluates
validation loss on its original epoch schedule (zero-based `epoch % 2 == 0`).
Merge selects exactly one best-validation-loss checkpoint, never test AP.

The opt-in `v2xset_camera_compatible: true` applies V2XSet's existing exclusion
of `2021_09_09_13_20_58` even to LiDAR-only inputs, on every split. With the flag
absent, existing dataset behavior is unchanged. This mirrors the current HEAL /
DOMA Camera-compatible scene policy; it is not a generic missing-image filter.

The isolated `lateheterbaseline` dataset requires LiDAR labels and explicitly
binds `generate_object_center_single` to the LiDAR generator. All detectors,
including cameras, receive LiDAR supervision. Validation/test CAVs retain base
sorting with infrastructure at the end; remapping every agent to a single
modality cannot change ego. Training retains existing shuffled CAV selection.

Final uses the same four-modality in-order assignment and ego m1 as HEAL ordered
inference. Evaluation GT is generated from every max_cav/communication-eligible
member directly in clean ego coordinates, deduplicated by ID and processed by
the existing GT postprocessor. It is independent of prediction use_cav. Final
range is [-204.8,204.8] x [-102.4,102.4], with original z limits [-3,1]. The same
existing NMS and AP@0.3/0.5/0.7 functions are used. No Fusion predicts only ego
through the strictly merged m1 branch. Report it once; any repeated table columns
must be labelled "single-vehicle baseline (same result)".

## Preflight (run once, before training)

Ensure the local change has been reviewed and deployed via GitHub first. The
runtime needs CUDA (the existing LSS constructor creates a small CUDA frustum),
the original OpenCOOD compiled extensions (including box_overlaps),
data under dataset/V2XSET/{train,validate,test}, and pretrained EfficientNet-B0
in the Torch cache or a permitted download route. Model construction invokes the
repository's existing EfficientNet pretrained initialization.

```bash
cd /home/fkk/my_data/disk2/code/DOMA
python -B -m opencood.tools.check_v2xset_baselines --models
python -B -m opencood.tools.v2xset_baselines assignments
python -B -m opencood.tools.v2xset_baselines audit-data --samples 3
mkdir -p opencood/logs/V2XSET_BASELINES_SEED303/console
```

Assignments reuse the existing generators. Existing files are validated and
never overwritten; a conflicting assignment fails. `audit-data` compares frame
and ego ordering across all five configurations on all splits, then checks the
first three test GT tensors against the actual HEAL infer dataset. Increase
`--samples` to the dataset length for full GT parity before formal comparison.
It reads data on CPU and does not build/run a detector. Static success alone
does not establish actual dataset or GT parity.

## Round 1: two existing screen sessions, foreground only

GPU 0 screen:

```bash
cd /home/fkk/my_data/disk2/code/DOMA
set -o pipefail
run_stamp=$(date +%Y%m%d_%H%M%S_%N)
python -B -m opencood.tools.v2xset_baselines prepare --modality m1 && \
CUDA_VISIBLE_DEVICES=0 python -u -m opencood.tools.train \
  --hypes_yaml opencood/hypes_yaml/v2xset/Baselines/m1_pointpillar.yaml \
  --model_dir opencood/logs/V2XSET_BASELINES_SEED303/single/m1_pointpillar \
  --fusion_method late 2>&1 | tee -a "opencood/logs/V2XSET_BASELINES_SEED303/console/m1_${run_stamp}.log"
```

GPU 2 screen:

```bash
cd /home/fkk/my_data/disk2/code/DOMA
set -o pipefail
run_stamp=$(date +%Y%m%d_%H%M%S_%N)
python -B -m opencood.tools.v2xset_baselines prepare --modality m2 && \
CUDA_VISIBLE_DEVICES=2 python -u -m opencood.tools.train \
  --hypes_yaml opencood/hypes_yaml/v2xset/Baselines/m2_lss_effnet.yaml \
  --model_dir opencood/logs/V2XSET_BASELINES_SEED303/single/m2_lss_effnet \
  --fusion_method late 2>&1 | tee -a "opencood/logs/V2XSET_BASELINES_SEED303/console/m2_${run_stamp}.log"
```

`prepare` creates a fresh directory containing only config.yaml and refuses an
existing directory. Passing it as model_dir starts the existing trainer at epoch
0 with no checkpoint; setup_lr_schedular treats 0 identically to a fresh start.
Do not reuse prepare as an implicit resume command.

## Round 2: only after BOTH round-1 processes complete

GPU 0 screen:

```bash
cd /home/fkk/my_data/disk2/code/DOMA
set -o pipefail
run_stamp=$(date +%Y%m%d_%H%M%S_%N)
python -B -m opencood.tools.v2xset_baselines prepare --modality m3 && \
CUDA_VISIBLE_DEVICES=0 python -u -m opencood.tools.train \
  --hypes_yaml opencood/hypes_yaml/v2xset/Baselines/m3_second.yaml \
  --model_dir opencood/logs/V2XSET_BASELINES_SEED303/single/m3_second \
  --fusion_method late 2>&1 | tee -a "opencood/logs/V2XSET_BASELINES_SEED303/console/m3_${run_stamp}.log"
```

GPU 2 screen:

```bash
cd /home/fkk/my_data/disk2/code/DOMA
set -o pipefail
run_stamp=$(date +%Y%m%d_%H%M%S_%N)
python -B -m opencood.tools.v2xset_baselines prepare --modality m4 && \
CUDA_VISIBLE_DEVICES=2 python -u -m opencood.tools.train \
  --hypes_yaml opencood/hypes_yaml/v2xset/Baselines/m4_lss_resnet101.yaml \
  --model_dir opencood/logs/V2XSET_BASELINES_SEED303/single/m4_lss_resnet101 \
  --fusion_method late 2>&1 | tee -a "opencood/logs/V2XSET_BASELINES_SEED303/console/m4_${run_stamp}.log"
```

## Merge after all four independent training jobs finish

```bash
python -B -m opencood.tools.v2xset_baselines merge
```

Defaults select the unique `net_epoch_bestval_at*.pth` in each listed single
directory. Explicit paths are available as `--m1 PATH --m2 PATH --m3 PATH --m4 PATH`.
Only exact independent-baseline configs and strictly loadable single states are
accepted. Keys must belong to their branch; unknown/missing/wrong-shape/wrong-dtype
keys fail. Any shared tensor must be equal; conflicts cannot silently overwrite.
The union is loaded strict=True into Final, serialized and strictly reloaded.

Created, without overwriting any existing output directory:

- `final_infer/config.yaml`
- `final_infer/net_epoch1.pth` (1 is a loader-compatible filename ID, not training epoch)
- `final_infer/sources.json` (source paths, source checkpoint/config SHA-256,
  Final checkpoint/config SHA-256, strict-load status)

## Final inference (user launches after reviewing training and parity checks)

No Fusion, exactly once:

```bash
set -o pipefail
run_stamp=$(date +%Y%m%d_%H%M%S_%N)
CUDA_VISIBLE_DEVICES=0 python -u -m opencood.tools.v2xset_baselines infer \
  --fusion_method no 2>&1 | tee -a "opencood/logs/V2XSET_BASELINES_SEED303/console/no_${run_stamp}.log"
```

Late Fusion, all four member counts sequentially:

```bash
set -o pipefail
run_stamp=$(date +%Y%m%d_%H%M%S_%N)
CUDA_VISIBLE_DEVICES=0 python -u -m opencood.tools.v2xset_baselines infer \
  --fusion_method late 2>&1 | tee -a "opencood/logs/V2XSET_BASELINES_SEED303/console/late_${run_stamp}.log"
```

Alternatively evaluate one member count per foreground command:

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m opencood.tools.v2xset_baselines infer --fusion_method late --use_cav 1
CUDA_VISIBLE_DEVICES=0 python -u -m opencood.tools.v2xset_baselines infer --fusion_method late --use_cav 2
CUDA_VISIBLE_DEVICES=0 python -u -m opencood.tools.v2xset_baselines infer --fusion_method late --use_cav 3
CUDA_VISIBLE_DEVICES=0 python -u -m opencood.tools.v2xset_baselines infer --fusion_method late --use_cav 4
```

Use either the all-count command or the four individual commands, not both for
the same output directory. Each result directory must not already exist. The
same Final config and strictly loaded checkpoint are used by both methods.
No Fusion reuses m1 weights from the merged checkpoint and ignores other branches
for prediction. This preserves full four-modality data/GT while running only ego.

Results under `opencood/logs/V2XSET_BASELINES_SEED303/final_infer/`:

| Experiment | AP file | Effective protocol / completion evidence |
|---|---|---|
| No Fusion | `no_fusion/eval.yaml` | `no_fusion/protocol.json`, `no_fusion/completed.json` |
| Late 1 | `late_use_cav1/eval.yaml` | same two JSON filenames in `late_use_cav1/` |
| Late 2 | `late_use_cav2/eval.yaml` | same in `late_use_cav2/` |
| Late 3 | `late_use_cav3/eval.yaml` | same in `late_use_cav3/` |
| Late 4 | `late_use_cav4/eval.yaml` | same in `late_use_cav4/` |

AP YAML keys follow the existing evaluator: `ap30`, `ap_50`, `ap_70`.
Protocol records actual dataset length, processed sample/GT counts, LiDAR GT,
range, comm_range, max_cav, ego, assignment/checkpoint/config hashes and method.
Only completed.json indicates the full requested evaluation finished.

## Local acceptance and publication

Local construction checks: five models construct; detection parameters are moved
to CPU for synthetic per-modality and combined strict loads. The unchanged camera
constructor allocates a small CUDA frustum, so `--models`, merge and infer require
CUDA even though merge/strict-load tensor operations use CPU. Malformed merges
are rejected; opt-in/default filtering and
GT function binding are exercised with stub parents. No forward is executed.
Real-data audit and trained checkpoint merge remain server validation tasks.
The local Windows environment lacks `opencood.utils.box_overlaps`, so the full
dataset import is not validated there. Do not treat CPU/static checks as CUDA/AP
acceptance or proof that V100 32 GB fits the original batch sizes.

Publication whitelist (review diffs, then explicitly add these paths only):

```bash
git add -- opencood/hypes_yaml/v2xset/Baselines/m1_pointpillar.yaml opencood/hypes_yaml/v2xset/Baselines/m2_lss_effnet.yaml opencood/hypes_yaml/v2xset/Baselines/m3_second.yaml opencood/hypes_yaml/v2xset/Baselines/m4_lss_resnet101.yaml opencood/hypes_yaml/v2xset/Baselines/final.yaml opencood/data_utils/datasets/__init__.py opencood/data_utils/datasets/basedataset/v2xset_basedataset.py opencood/data_utils/datasets/late_heter_baseline_fusion_dataset.py opencood/tools/v2xset_baselines.py opencood/tools/check_v2xset_baselines.py docs/V2XSET_BASELINES.md
git diff --cached --check
git diff --cached --name-only
```

Do not include train.py/train_ddp.py's existing changes, DAIR, Table4, scripts,
logs, generated assignments, checkpoints or results in this change. No Git write
operation or server command was run during local implementation.
