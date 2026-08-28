# DOMA Clean Design

## 1. Scope and invariants

DOMA (Dual-Space Object-Centric Multi-Granularity Alignment) is implemented as
an additive model family on top of the Official HEAL code at
`96812ed5a41619bebb2af4f84b803b7b7468d1d3`. Official HEAL YAMLs continue to
resolve to Official models, losses, fusion, training, merge, and inference
code. Only YAMLs under `MoreModality/DOMA` enter the DOMA path.

The implementation follows four non-negotiable invariants:

1. The Official training driver is untouched. No accumulator, AMP wrapper,
   scaler, seed mutation, post-training hook, or zero-gradient reordering is
   introduced.
2. `doma.version` is a label and a profile validator. Runtime tensor operations
   are selected by explicit YAML switches.
3. A disabled module is not instantiated. It creates no parameters, consumes
   no initialization RNG, enters no forward branch, and contributes no loss.
4. V1--V3 use one Python implementation. Later versions are strict module
   increments rather than copied model files.

The isolation boundary is:

```text
Official YAML -> Official model/loss/fusion -> Official train/inference

DOMA YAML     -> DOMA model/loss/fusion     -> Official train driver
                                             + DOMA-only merge/inference helper
```

## 2. Method structure

HEAL supplies feature-space alignment: each modality encoder/backbone is
aligned to Common BEV, warped to the ego frame, and fused by the pyramid
backbone. DOMA adds a second, object-centric space after that alignment.

For a proposal `b` and agent `a`, DOMA samples a rotated ROI from the aligned
Common-BEV feature. The detail ROI is adapted by a modality-owned residual
1x1 adapter and encoded by a Stage1-owned shared object encoder. Proposal
geometry is normalized and encoded by a shared geometry encoder. A shared
zero-output-initialized refiner predicts the periodic 8D box residual
`[dx, dy, dz, dlogl, dlogw, dlogh, sin(dyaw), cos(dyaw)-offset]`.

The version increments are:

- V1: detail ROI, object adapter, shared object encoder, geometry encoder,
  refiner, and uniform geometry consensus.
- V2: V1 plus a context ROI from the pre-fusion pyramid, a context adapter and
  encoder, and residual-safe detail/context concat projection.
- V3: V2 plus a quality head, refined-IoU quality supervision, and
  quality-weighted consensus. V3 is an **Experimental V3 extension**; this
  repository makes it reproducible but does not claim it is empirically
  effective.

The refiner and geometry encoder already belong to V1. They are not V3
features. Full historical evidence is recorded in
`docs/DOMA_VERSION_EVOLUTION.md`.

## 3. Explicit YAML contract

Every DOMA YAML has the complete `model.args.doma` mapping. The validator in
`doma_config.py` rejects ambiguous, incomplete, or cross-version profiles.

| Section | Explicit control | Effect |
|---|---|---|
| `enabled` | DOMA family gate | Must be true in DOMA YAMLs |
| `version` | `v1`, `v2`, `v3` | Label/profile consistency only |
| `ablation` | boolean | Permits explicit lower-feature combinations within the version ceiling |
| `mode` | `stage1_anchor`, `stage2_adapt`, `inference` | Trainability and data-flow protocol |
| `object_space.enabled` | boolean | Rotated object-space ROI branch |
| `object_adapter.enabled` | boolean | Per-modality residual adapter |
| `object_encoder.enabled` | boolean | Shared detail encoder |
| `geometry.enabled` | boolean | Shared normalized-geometry encoder |
| `refiner.enabled` | boolean | Shared residual box refiner |
| `multi_granularity.enabled` | boolean | Detail/context representation path |
| `multi_granularity.detail/context/fusion.enabled` | booleans | Individual V2 submodule gates |
| `quality.enabled` | boolean | V3 quality head and quality loss payload |
| `consensus.enabled` and `mode` | boolean/mode | Uniform or quality-weighted geometry consensus |
| `training_proposals.source` | `gt_jitter` | The only supported V1--V3 training proposal source |
| `loss.*_weight` | non-negative scalars | Individual, consensus, and quality terms |

Canonical method-switch profiles are checked exactly; numeric settings are
strictly type/range checked, while the packaged YAML values are separately
locked by per-version SHA-256 method fingerprints, recursive Official-parity,
and count tests. For an ablation,
`ablation: true` is required, and the version ceiling still prevents V1 from
enabling V2/V3 modules or V2 from enabling the V3 quality module. A V3
without quality must explicitly replace `quality` with `enabled: false`,
change consensus to uniform (or disable it), and remove quality-only fallback
fields. No Python edit is required.

## 4. Random-initialization isolation

`install_doma_modules` constructs modules in historical order and only inside
their feature gate:

```text
V1 core: ROI/sampler -> adapters -> object encoder -> geometry -> refiner
V2 add:  context ROI/adapters -> context encoder -> multigranularity fusion
V3 add:  quality head
```

There are no dormant V2 objects in V1 and no dormant V3 quality object in
V1/V2. The local static acceptance check asserts both attribute absence and
state-key absence. This protects initialization order as well as forward
behavior.

The adapters and refiner are identity-safe at construction: the last adapter
convolution and the refiner output layer are zero initialized. The centered
yaw residual additionally makes a zero residual an exact identity for V2/V3.

## 5. Stage1 and Stage2 protocol

### Stage1 anchor

Stage1 retains the Official collaborative detector and trains the shared DOMA
anchor using modality m1. Because m1 defines the anchor space, it uses the
identity path and has no registered m1 adapter. DOMA-shared
encoders/refiner/fusion/quality modules are trainable. Training proposals come
only from ground truth plus controlled jitter; predicted or mixed proposals
are intentionally out of scope.

### Stage2 adaptation

Stage2 retains Official HEAL encoder/backbone/aligner and backward-alignment
training. The Stage1-owned DOMA shared modules are loaded through Official
`strict=False` checkpoint loading and frozen. Only the active modality's object
adapter is trainable; V2/V3 also train that modality's context adapter.

DOMA-specific trainable parameter counts in Stage2 are:

| Version | Active DOMA trainable parameters |
|---|---:|
| V1 | 8,448 |
| V2 | 41,728 |
| V3 | 41,728 |

No optimizer, scheduler, epoch, batch-size, validation, backward, or step
logic is changed. Those decisions remain entirely in Official `train.py`.

## 6. Forward and loss

The detector forward is the Official HEAL forward copied into a separate DOMA
class, with the following additive payload:

1. Preserve aligned per-agent Common-BEV features and transforms.
2. For V2/V3, ask the DOMA pyramid subclass for the pre-fusion context level.
3. During training, sample GT+jitter proposals, extract valid agent/object
   ROIs, predict individual residuals, then form consensus residuals.
4. Attach the result as `output_dict['doma_object']`.

`DOMAPointPillarPyramidLoss` calls `PointPillarPyramidLoss.forward` first, then
adds `compute_doma_object_loss` only when that payload exists. The base loss
therefore retains Official arithmetic. The DOMA loss can contain:

- individual Smooth-L1 residual loss;
- consensus Smooth-L1 residual loss;
- V3-only Smooth-L1 quality loss against detached refined IoU.

V1 and V2 have no quality head, quality tensor, or quality loss. The loss
weights are explicit in YAML.

## 7. Fusion behavior

`DOMAPyramidFusion` subclasses Official `PyramidFusion` and defines no
constructor, so its parameters and initialization order are identical.

- When pre-fusion features are not requested, it calls Official
  `PyramidFusion.forward_single` or `forward_collab` directly. This is the V1
  path.
- V2/V3 request the selected pre-fusion pyramid level in addition to the
  normal fused output. That level feeds only the context ROI branch.

Official `pyramid_fuse.py` is unchanged and Official HEAL never imports the
DOMA subclass.

## 8. Checkpoints and merge ownership

Official `load_state_dict(..., strict=False)` behavior is preserved. No legacy
checkpoint validator is migrated.

`doma_tools.py merge_final` follows Official m2, m3, m4, m1 merge order, then
enforces explicit DOMA ownership:

| Keys | Owner |
|---|---|
| `doma_shared_object_encoder.*` | Stage1 m1 |
| `doma_shared_geometry_encoder.*` | Stage1 m1 |
| `doma_shared_object_refiner.*` | Stage1 m1 |
| `doma_shared_context_encoder.*` | Stage1 m1, V2/V3 only |
| `doma_shared_multigranularity_fusion.*` | Stage1 m1, V2/V3 only |
| `doma_shared_quality_head.*` | Stage1 m1, V3 only |
| `doma_object_adapter_m2/m3/m4.*` | Matching Stage2 checkpoint |
| `doma_context_adapter_m2/m3/m4.*` | Matching Stage2 checkpoint, V2/V3 only |

Before writing, the tool checks that all four configs have the same method
fingerprint and that every Stage2 copy of frozen shared DOMA tensors is exactly
equal to Stage1. Each owned adapter must also contain exactly its six expected
parameter keys; partial or extra adapter state fails closed.

## 9. Inference behavior

`inference_doma_heter_in_order.py` preserves the Official ordered-modality
dataset and postprocessor flow. It supports the same intermediate fusion mode,
then applies DOMA refinement after Official postprocessing:

- take at most the top 64 post-NMS proposals;
- refine only proposals with valid object-space evidence;
- leave classification scores unchanged;
- do not create proposals and do not run a second NMS;
- fall back to the original box when evidence is invalid.

V1/V2 use uniform valid-agent geometry consensus. V3 uses detached predicted
quality weights and falls back to the uniform mean when their sum is too low.

## 10. File layout

```text
opencood/models/
  doma_heter_pyramid_collab.py
  doma_heter_pyramid_single.py
  fuse_modules/doma_pyramid_fuse.py
  sub_modules/doma_config.py
  sub_modules/doma_box_coder.py
  sub_modules/doma_object_roi.py
  sub_modules/doma_proposal_sampler.py
  sub_modules/doma_object.py
opencood/loss/
  doma_object_loss.py
  doma_point_pillar_pyramid_loss.py
opencood/tools/
  doma_tools.py
  inference_doma_heter_in_order.py
  check_doma_static.py
  check_doma_functional.py
  check_doma_model_counts.py
  check_doma_merge.py
opencood/hypes_yaml/opv2v/MoreModality/DOMA/
  V1/{stage1,stage2,final_infer}
  V2/{stage1,stage2,final_infer}
  V3/{stage1,stage2,final_infer}
```

Deliberately absent are the old engineering/training-diagnostics framework,
AMP/accumulation additions, post-training automation, predicted/mixed
proposals, RPR, and later experimental V4--V6 modules.
