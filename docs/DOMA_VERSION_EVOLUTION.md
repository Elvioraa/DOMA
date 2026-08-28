# DOMA Version Evolution

## 1. Audit basis

The old repository `F:\zzuStudy\exp_gpt\HEAL-XLab` was audited read-only with
file search plus `git log`, `git show`, `git diff`, and `git blame`. The clean
target repository starts at Official HEAL commit
`96812ed5a41619bebb2af4f84b803b7b7468d1d3`.

Important historical evidence:

| Commit | Evidence |
|---|---|
| `96812ed5a41619bebb2af4f84b803b7b7468d1d3` | Official HEAL baseline; no object-space implementation |
| `12218ef5ce713f370e4583214e7a613eb9c0b120` | Introduced the old V1/V2/V3 implementation and YAML profiles together |
| `0f3acf92e2c977d9600dee7506f1434c92475676` | Corrected device/dtype handling in geometry code; absorbed as an implementation fix |
| `840b36afd7d0f0d01afda06116ccae00ea30397e` | Old **DS V1.1** centered-yaw identity fix: `cos(dyaw)` became `cos(dyaw)-1` |
| `3da8ea794cec16a7413841e13bdf8bdd032a763e` | Only `ds_v2_1` hit is a synthetic smoke-profile label, not a method version |
| `e3ac402` | Documentation's “HEAL-XLab-v2.1” refers to HVP-CBEA training-only work, not DOMA V2.1 |

The old V1/V2/V3 labels were configuration profiles introduced in one commit,
not three chronological implementation commits. Their incremental meaning is
nevertheless explicit in code and YAML and is preserved below.

## 2. Audited historical matrix

`ON` means the old profile instantiated and used the component. “No version”
means no distinct V2.1 YAML/model/loss/checkpoint/merge protocol existed.

| Component | Official HEAL | Old V1 | Old V2 | Old V2.1 | Old V3 |
|---|---:|---:|---:|---:|---:|
| Feature alignment | ON | ON | ON | No version | ON |
| Object space | OFF | ON | ON | No version | ON |
| Object ROI | OFF | Detail 5x5 | Detail 5x5 | No version | Detail 5x5 |
| Object encoder | OFF | ON | ON | No version | ON |
| Geometry encoder | OFF | ON | ON | No version | ON |
| Object adapter | OFF | ON | ON | No version | ON |
| Context adapter | OFF | OFF | ON | No version | ON |
| Detail ROI | OFF | ON | ON | No version | ON |
| Context ROI | OFF | OFF | 3x3 | No version | 3x3 |
| Multi-granularity fusion | OFF | OFF | Concat projection | No version | Concat projection |
| Refiner | OFF | ON | ON | No version | ON |
| Quality head | OFF | OFF | OFF | No version | ON |
| Quality loss | OFF | OFF | OFF | No version | Refined-IoU Smooth-L1 |
| Quality consensus | OFF | OFF | OFF | No version | Quality weighted |
| Consensus | OFF | Uniform valid-agent mean | Uniform valid-agent mean | No version | Quality weighted with uniform fallback |
| Historical yaw encoding | - | `sin_cos` | `sin_cos` | No version | `sin_cos` |

The refiner was present in old V1. V3 did not introduce proposal creation,
score recalibration, or a second NMS. Later predicted-proposal and RPR variants
belong to post-V3 work and are excluded.

## 3. V2 versus V2.1 decision

There is **no formal DOMA V2.1**.

- `ds_v2_1` appears only as a synthetic inference smoke-test label whose
  effective configuration is V2 (`multi=True`).
- The separately named “HEAL-XLab-v2.1” history is HVP-CBEA work and is outside
  the DOMA method boundary.
- The real intermediate fix is DS V1.1: centered cosine residual encoding plus
  zero-output initialization makes a zero predicted residual an exact identity.
  It changes no module, parameter, state key, merge owner, or loss family.

DOMA therefore exposes V1, V2, and V3 only. Strict historical V1 reproduction
keeps legacy `sin_cos`. The centered identity correction is absorbed into the
canonical DOMA V2 and retained in V3 as `sin_cos_centered`. This is a deliberate
correctness decision, not a claim that old production V2/V3 used the centered
encoding. The old production V2/V3 profiles used `sin_cos`.

## 4. Clean DOMA evolution matrix

| Component | HEAL | DOMA V1 | DOMA V2 | DOMA V3 |
|---|---:|---:|---:|---:|
| Feature alignment | ON | ON | ON | ON |
| Object-space detail ROI | OFF | ON | ON | ON |
| Stage2 per-modality object adapter | OFF | ON | ON | ON |
| Shared object encoder | OFF | ON | ON | ON |
| Shared geometry encoder | OFF | ON | ON | ON |
| Shared refiner | OFF | ON | ON | ON |
| Uniform geometry consensus | OFF | ON | ON | Fallback |
| Context ROI/adapter/encoder | OFF | OFF | ON | ON |
| Detail-context fusion | OFF | OFF | ON | ON |
| Quality head/loss | OFF | OFF | OFF | ON |
| Quality-weighted consensus | OFF | OFF | OFF | ON |
| Yaw mode | - | `sin_cos` | `sin_cos_centered` | `sin_cos_centered` |
| Status | Official | Canonical | Canonical + absorbed fix | Experimental V3 extension |

## 5. Exact incremental contracts

### HEAL to DOMA V1

- Modules: rotated detail ROI sampler, GT+jitter proposal sampler,
  modality-owned residual object adapters, shared object encoder, shared
  geometry encoder, shared zero-output refiner.
- Parameters: 138,248 Stage1 DOMA parameters; 163,592 in the final four-modality
  DOMA overlay.
- Forward: construct per-agent object embeddings and residuals; uniform mean
  across valid agent observations.
- Loss: individual plus consensus residual Smooth-L1.
- Checkpoint/merge: Stage1 owns shared modules; each Stage2 owns its adapter.
- Inference: refine up to 64 post-NMS boxes, preserve scores, no second NMS.
- YAML: explicit core object/geometry/refiner/consensus switches.

### DOMA V1 to DOMA V2

- Modules: context ROI sampler, modality-owned context adapters, shared context
  encoder, shared detail-context concat projection.
- Parameters: +177,281 Stage1 DOMA parameters and +277,121 final-overlay DOMA
  parameters relative to V1.
- Forward: capture a pre-fusion pyramid context feature, sample 3x3 context ROI,
  and fuse it with the 5x5 detail representation.
- Loss: same residual loss family; no quality term.
- Geometry: same encoder/refiner, with centered yaw identity fix in clean DOMA.
- Merge: adds Stage1-owned context encoder/fusion and Stage2-owned context
  adapters.
- Inference: same proposal and postprocessing contract, now with two-granularity
  evidence.
- YAML: explicitly enables detail, context, and fusion; quality remains false.

### DOMA V2 to DOMA V3

- Modules: shared object quality head.
- Parameters: +10,497 parameters and +4 state tensors in both Stage1 and final
  overlay.
- Forward: predict per-agent quality from object and geometry embeddings, ROI
  coverage, and normalized agent distance; use detached weights for consensus.
- Loss: add weight-0.05 Smooth-L1 against detached refined-IoU target.
- Merge: quality head is Stage1-owned; no new Stage2-owned parameters.
- Inference: quality-weighted consensus with an explicit uniform fallback.
- YAML: explicitly enables the quality head and quality consensus.

V3 is retained for reproducibility and ablation. The old history contains no
tracked evidence sufficient to claim that quality improves AP or avoids
collapse, so no effectiveness claim is made here.

## 6. Parameters, tensors, and isolation evidence

Counts below are exact for the DOMA-specific module set and are produced by
`python -B -m opencood.tools.check_doma_static`.

| Version | Stage1 DOMA params | Stage1 state tensors | Final-overlay DOMA params | Final-overlay state tensors |
|---|---:|---:|---:|---:|
| V1 | 138,248 | 24 | 163,592 | 42 |
| V2 | 315,529 | 41 | 440,713 | 77 |
| V3 | 326,026 | 45 | 451,210 | 81 |

The Official m1 Stage1 model has 5,464,791 parameters, all trainable, and 382
state tensors. Because the DOMA model retains the same base and adds only the
listed modules, full Stage1 counts are:

| Model | Parameters | Trainable parameters | State tensors | DOMA-specific modules |
|---|---:|---:|---:|---|
| Official HEAL | 5,464,791 | 5,464,791 | 382 | None |
| DOMA V1 Stage1 | 5,603,039 | 5,603,039 | 406 | ROI/sampler, object encoder, geometry encoder, refiner |
| DOMA V2 Stage1 | 5,780,320 | 5,780,320 | 423 | V1 + context ROI/encoder + multi-granularity fusion |
| DOMA V3 Stage1 | 5,790,817 | 5,790,817 | 427 | V2 + quality head |

V1 has no context or quality attributes/state keys. V2 has context attributes
but no quality attributes/state keys. This is checked structurally, not inferred
from whether a forward happened to use a pre-created module.

Stage1 m1 is the identity anchor and registers no m1 adapter. The exact
DOMA-only Stage2/final-mode counts are:

| Mode | V1 params/state/trainable | V2 params/state/trainable | V3 params/state/trainable |
|---|---:|---:|---:|
| One Stage2 modality | 146,696 / 30 / 8,448 | 357,257 / 53 / 41,728 | 367,754 / 57 / 41,728 |
| Four-modality merged inference | 163,592 / 42 / 0 | 440,713 / 77 / 0 | 451,210 / 81 / 0 |

## 7. YAML control matrix

All five files per version use the same method fingerprint; only `mode` and
the Stage2 `active_modality` differ.

| YAML control | V1 | V2 | V3 |
|---|---|---|---|
| `object_space.enabled` | true | true | true |
| `object_adapter.enabled` | true | true | true |
| `object_encoder.enabled` | true | true | true |
| `geometry.enabled` | true | true | true |
| `refiner.enabled` | true | true | true |
| `multi_granularity.enabled` | false | true | true |
| `detail.enabled` | absent | true | true |
| `context.enabled` | absent | true | true |
| `fusion.enabled` | absent | true | true |
| `quality.enabled` | false | false | true |
| `consensus.mode` | uniform | uniform | quality weighted |
| `refiner.yaw_mode` | `sin_cos` | `sin_cos_centered` | `sin_cos_centered` |
| `quality_loss_weight` | 0 | 0 | 0.05 |

Each version contains `stage1/m1.yaml`, `stage2/m2.yaml`, `stage2/m3.yaml`,
`stage2/m4.yaml`, and `final_infer/m1m2m3m4.yaml`. After stripping the DOMA
block and normalizing only model/loss names, all 15 files are semantically equal
to their Official HEAL counterparts for dataset, anchors, postprocess,
batch size, epochs, optimizer, learning rate, scheduler, and validation fields.
