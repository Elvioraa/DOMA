# DAIR-V2X-C local adaptation

Reviewable local implementation, not evidence of Silas/CUDA/AP parity.
Dataset geometry, ROI, box coder, consensus and Object Refiner are unchanged.

## Configuration inventory

Root: `opencood/hypes_yaml/dairv2x/MoreModality/DAIR_PROTOCOL/`.
Both `HEAL/` and `DOMA/` contain seven `config.yaml` files:

- `stage1/`
- `E1/stage2_m2/`, `E1/final_m1m2/`
- `E2/stage2_m2/`, `E2/final_m1m2/`
- `E3/stage2_m2/`, `E3/final_m1m2/`

E1=EfficientNet-LSS, E2=SECOND, E3=existing ResNet-LSS.
Stage2 always uses m2. Final Vehicle=m1, Infrastructure=m2. The original
100 m communication rule can filter a collaborator; no nonexistent agent
is forced into a sample.

The DOMA method block matches the existing full V2R2 snapshot and the tracked
`DOMA/V2/stage1/m1.yaml` method block exactly, apart from later-stage
mode/active_modality. Runtime and tests do not depend on untracked Table4 files.

Protocol: seed 303, train/validation 4811/1789, communication 100 m, range
[-102.4,-51.2,-3.5,102.4,51.2,1.5]. Training batch 2 per process, 4 on two GPUs.
Stage1: Adam .002, 40 epochs, milestones [15,30]. Stage2: Adam .001,
30 configured epochs, milestones [15,25]. eps=1e-10. Base decay=1e-4;
DOMA decay=0 explicitly preserves the existing V2R2 method. Both optimizer
groups use the same stage LR. HEAL has only its Base group.

Stage2 net_epoch1 initialization retains indices 1..29, not 30 additional epochs.
eval/save frequency remains 2; Stage2 validation indices are 2,4,..,28.
Safe DDP uses global sum/count, nonpadding 895/894 validation shards and rank0
checkpoint writes. DDP validation batch remains 1. Training sampler pads
4811 to 4812 globally. bestval selects minimum loss, not AP.

Stage2 keeps label_type=camera, including SECOND, and native random
physical-side selection. Final uses intermediateheter/lidar GT, score .2/NMS
.15, and retains the original Stage2 versus Final anchor differences.

## Explicit source and output ownership

No command chooses historical Stage1 weights automatically. A verified Stage1
may serve several Stage2 runs, or each experiment may train its own Stage1;
this remains a user decision. Never use OPV2V-H or pure HEAL as complete DOMA
Stage1. Suggested output layout:

`opencood/logs/dairv2x_c/<HEAL|DOMA>/<run-id>/<stage1|E1|E2|E3>/...`

`dair_workflow prepare` refuses existing output directories. Stage1 preparation
copies config only and rejects source weights. Stage2 requires explicit source
checkpoint/config, builds real source/target models, validates strict source
schema and allowed inheritance, and copies the source as net_epoch1.pth.
stage1_source.json records absolute paths, SHA256 hashes and config snapshot.

At load time hashes are rechecked; Stage2 config edits fail closed. Only new m2
frontend/Adapter keys may be missing, and only m1 frontend keys may be extra.
Shared detection/DOMA keys, shapes/dtypes and BN buffers must match. Later
Stage2 checkpoints load strictly and must retain identical shared tensors.
Restoration is weights-only: Adam moments, RNG and historical best-loss state
are NOT restored. Original epoch selection is unchanged. Schema validation
does not prove training completion: select a verified Stage1 bestval with logs.

`dair_workflow merge` requires an explicit trained Stage2 checkpoint, checks
provenance, shared equality and Final config, and strictly loads the real Final
model before writing a fresh output directory plus merge_sources.json.
m1/shared tensors come from Stage1; m2 frontend/Adapters come from Stage2.
The existing four-checkpoint API remains available.

## Local checks

```text
python -B -m opencood.tools.check_dair_adaptation
python -B -m opencood.tools.check_doma_static
python -B -m opencood.tools.check_doma_functional
python -B -m opencood.tools.check_doma_merge
python -B -m opencood.tools.check_doma_optimizer_groups
python -B -m opencood.tools.check_ddp_safe_validation
```

The DAIR component test uses real DOMA modules, synthetic BEV tensors and small
checkpoint fixtures. It is NOT a complete encoder/detector or real-data test.
Existing model-count/DQC checks may use optional-import stubs; their PASS does
not establish real encoder operation.

## Silas short tests AFTER review and approved deployment

Instructions only; not executed remotely. The environment must already have
dependencies and the EfficientNet pretrained cache. Without the cache, upstream
model construction may attempt downloading weights. Nothing is installed here.

Stage1 short tests: two optimizer steps, two validation batches per rank,
temporary rank0 checkpoint save/reload and cleanup; no persistent checkpoint.

```bash
(
set -eu
cd /home/fkk/my_data/disk2/code/DOMA || exit 1
PY=/home/fkk/my_data/disk1/miniconda3/envs/DOMA/bin/python
CFG=opencood/hypes_yaml/dairv2x/MoreModality/DAIR_PROTOCOL
for METHOD in HEAL DOMA; do
  CUDA_VISIBLE_DEVICES=0,2 "$PY" -B -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=2 \
    -m opencood.tools.check_dair_runtime \
    --config "$CFG/$METHOD/stage1/config.yaml" --num-workers 0
done
)
```

Stage2 requires deliberately selected same-method DAIR Stage1 sources.
Set absolute DAIR_STAGE1_CHECKPOINT and DAIR_STAGE1_CONFIG plus DAIR_METHOD
(HEAL or DOMA). The short Stage1 test deletes its checkpoint and cannot supply
a trained Stage1 for experiments.

```bash
(
set -eu
cd /home/fkk/my_data/disk2/code/DOMA || exit 1
PY=/home/fkk/my_data/disk1/miniconda3/envs/DOMA/bin/python
: "${DAIR_STAGE1_CHECKPOINT:?Select a verified absolute Stage1 checkpoint}"
: "${DAIR_STAGE1_CONFIG:?Select its absolute config.yaml}"
: "${DAIR_METHOD:?Set HEAL or DOMA}"
case "$DAIR_METHOD" in HEAL|DOMA) ;; *) exit 1 ;; esac
CFG=opencood/hypes_yaml/dairv2x/MoreModality/DAIR_PROTOCOL
for E in E1 E2 E3; do
  CUDA_VISIBLE_DEVICES=0,2 "$PY" -B -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=2 \
    -m opencood.tools.check_dair_runtime \
    --config "$CFG/$DAIR_METHOD/$E/stage2_m2/config.yaml" \
    --stage1-checkpoint "$DAIR_STAGE1_CHECKPOINT" \
    --stage1-config "$DAIR_STAGE1_CONFIG" --num-workers 0
done
)
```

Logical GPUs 0/1 map to visible physical 0/2. Verify world_size=2,
global_batch=4, two steps, finite losses, frozen equality, expected updates and
successful exit. Observed Agent counts are printed; unobserved A=1/2 cases are
not claimed as covered. Short validation is not formal bestval or AP.

## Preparation, merge, inference and counts interfaces

`python -B -m opencood.tools.dair_workflow --help` exposes:

```text
prepare --config CONFIG --output NEW_RUN_DIRECTORY
prepare --config STAGE2_CONFIG --output NEW_STAGE2_DIRECTORY --stage1-checkpoint ABS_PTH --stage1-config ABS_CONFIG
merge --config FINAL_CONFIG --stage2-dir STAGE2_DIRECTORY --stage2-checkpoint ABS_STAGE2_BESTVAL --output NEW_FINAL_DIRECTORY
counts --stage1 STAGE1_CONFIG --stage2 STAGE2_CONFIG --final FINAL_CONFIG
```

Future training uses the existing trainer with the prepared --model_dir.
No command in this document starts a full 40/30-epoch experiment.

Final: `python -B -m opencood.tools.inference_dair --model_dir FINAL_DIRECTORY`.
Optional --max-samples N writes clearly named subset results, never full-table
results. Both methods use the same Dataset, postprocess and eval_utils.
DOMA reuses same-forward post-NMS refinement, with no second NMS/score change.
Full output: eval_validation.yaml keys ap30/ap_50/ap_70, plus protocol metadata.
This evaluates validation, not an independent test.

Counts require real model dependencies, no stubs. Output contains per-stage
total/trainable/frozen and Base/DOMA counts, Final count and raw two-stage sums.
paper_sigma_m_params remains null pending verification of the author's exact
per-stage counting boundary. HEAL Table 3 defines accumulation over base
training and each integration:
https://arxiv.org/html/2401.13964v2#S5.T3
Neither trainable-only sum nor Final count is silently labelled as the paper
metric. Do not multiply by epoch count, DDP replicas or physical agents.

## Compatibility boundary

Original HEAL/OPV2V YAML and core model/loss/geometry are untouched. DAIR
contracts require explicit configuration. Post-training ordinary inference
defaults to enabled for legacy callers and is disabled in DAIR YAML.
Shared trainers now expose model_train_init errors and close writer/process
groups; these lifecycle improvements also affect legacy callers, without
changing optimizer loops, scheduler, sampler or bestval. Full OPV2V CUDA parity
is not claimed.

