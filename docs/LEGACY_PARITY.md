# Legacy Training Parity

## 1. Reference

- Branch inspected: `doma-dev`
- Official baseline commit: `96812ed5a41619bebb2af4f84b803b7b7468d1d3`
- Baseline tag: `heal-official-base-96812ed`
- Initial worktree status: clean
- Old reference repository: `F:\zzuStudy\exp_gpt\HEAL-XLab` (read-only)

The parity objective is exact isolation: the same Official YAML continues to
import the same Official Python files and follows the same tensor, optimizer,
and data-loading path. DOMA is selected only by a new core method in a DOMA
YAML.

## 2. Source-parity gate

The following files are zero-diff against the baseline:

| Official file | Status |
|---|---|
| `opencood/tools/train.py` | UNCHANGED |
| `opencood/tools/train_ddp.py` | UNCHANGED |
| `opencood/tools/train_utils.py` | UNCHANGED |
| `opencood/models/heter_pyramid_collab.py` | UNCHANGED |
| `opencood/models/heter_pyramid_single.py` | UNCHANGED |
| `opencood/models/fuse_modules/pyramid_fuse.py` | UNCHANGED |
| `opencood/loss/point_pillar_pyramid_loss.py` | UNCHANGED |
| `opencood/tools/heal_tools.py` | UNCHANGED |
| `opencood/tools/inference_heter_in_order.py` | UNCHANGED |

Re-run the hard gate from the repository root:

```powershell
git diff --exit-code heal-official-base-96812ed -- `
  opencood/tools/train.py `
  opencood/tools/train_ddp.py `
  opencood/tools/train_utils.py
```

Expected result: exit code 0 and zero diff output. A nonzero result is a parity
failure unless explicitly approved.

The broader Official path gate is:

```powershell
git diff --exit-code heal-official-base-96812ed -- `
  opencood/models/heter_pyramid_collab.py `
  opencood/models/heter_pyramid_single.py `
  opencood/models/fuse_modules/pyramid_fuse.py `
  opencood/loss/point_pillar_pyramid_loss.py `
  opencood/tools/heal_tools.py `
  opencood/tools/inference_heter_in_order.py
```

## 3. What was not migrated

The new files contain no training accumulator, AMP/autocast/GradScaler wrapper,
seed mutation, post-training runner, safe-inference framework, NICS, SEM, CA,
RFG, PACT, HVP, bandwidth robustness, predicted/mixed proposal path, RPR,
or V4--V6 implementation.

In particular, the Official ordering remains:

```text
model.zero_grad()
optimizer.zero_grad()
forward -> loss -> backward
optimizer.step()
```

No deterministic flag or seed was added to Official source. Any fixed-state
test must run as a separate diagnostic process.

## 4. New-file boundary

Only DOMA-specific source and configuration files are added:

- two DOMA model entry points;
- one DOMA fusion subclass;
- five object/config/box/ROI/proposal submodules;
- one DOMA loss entry point and one object loss implementation;
- DOMA-only merge and ordered-inference tools;
- static, CPU functional, full-count, and merge diagnostic scripts;
- 15 DOMA YAMLs and three documents.

Official YAMLs are not edited and require no `doma.enabled: false` field.

## 5. Three-layer regression protocol

### A. Source parity

Run both zero-diff commands above. Also confirm `git status --short` contains
only new DOMA paths and documentation. This is the strongest local evidence
that training semantics are unchanged.

### B. Fixed-checkpoint inference parity

On the server, use one copy of the Official config and one exact Official
checkpoint for both the pre-change baseline checkout and this checkout's
Official path.

1. Pin the same CUDA device, environment, dataset assignment JSON, and command.
2. Run Official `inference_heter_in_order.py` in both checkouts.
3. Compare prediction artifacts when available, then compare AP30/AP50/AP70 at
   the original printed precision.
4. Require exact equality. “Close AP” is not sufficient for this gate.

Do not use a DOMA YAML or DOMA inference script for this Official parity test.

### C. Fixed-state one-step training parity

Run this only from an external diagnostic launcher, never by editing
`train.py`.

1. In the baseline checkout, serialize one already-collated batch before it is
   moved to CUDA. Save the checkpoint, optimizer state, Python/NumPy/Torch CPU
   RNG states, all CUDA RNG states, epoch, and learning rate.
2. Use the identical serialized inputs, checkpoint, optimizer state, GPU,
   PyTorch/CUDA environment, and restored RNG states in both checkouts.
3. Execute exactly the Official step order: model zero-grad, optimizer
   zero-grad, device transfer, forward, Official loss (including `_single` when
   configured), backward, optimizer step.
4. Save tensors keyed by name for forward output, scalar loss, gradients, and
   post-step parameters.
5. Compare shapes, dtypes, devices, key sets, then bitwise tensor equality.
   If the deployed CUDA kernel is known nondeterministic, first establish that
   limitation with two replays of the same checkout; do not relax the source
   gate.

Suggested artifact layout:

```text
parity_fixture/
  config.yaml
  batch_cpu.pth
  model_before.pth
  optimizer_before.pth
  rng_state.pth
baseline_capture/
  forward.pth
  loss.json
  gradients.pth
  model_after.pth
candidate_capture/
  forward.pth
  loss.json
  gradients.pth
  model_after.pth
```

## 6. Local acceptance commands

```powershell
python -B -m opencood.tools.check_doma_static
python -B -m opencood.tools.check_doma_functional
python -B -m opencood.tools.check_doma_model_counts
python -B -m opencood.tools.check_doma_merge
python -m py_compile `
  opencood/models/doma_heter_pyramid_collab.py `
  opencood/models/doma_heter_pyramid_single.py `
  opencood/models/fuse_modules/doma_pyramid_fuse.py `
  opencood/models/sub_modules/doma_config.py `
  opencood/models/sub_modules/doma_box_coder.py `
  opencood/models/sub_modules/doma_object_roi.py `
  opencood/models/sub_modules/doma_proposal_sampler.py `
  opencood/models/sub_modules/doma_object.py `
  opencood/loss/doma_object_loss.py `
  opencood/loss/doma_point_pillar_pyramid_loss.py `
  opencood/tools/doma_tools.py `
  opencood/tools/inference_doma_heter_in_order.py
```

The static check covers YAML parsing, strict schema validation, canonical
profiles and method fingerprints, version isolation, explicit ablations,
import-name resolution, DOMA parameter/state counts, Official-YAML semantic
parity, and the centered-yaw identity. The CPU
functional check executes object-space forward, loss, and backward for V1,
V2, and V3 on synthetic tensors, then verifies same-forward inference keeps
proposal count and scores while changing only valid refined boxes.

Local acceptance result at implementation time:

| Check | Result |
|---|---|
| Python syntax compile | PASS |
| 15/15 YAML parse and strict validation | PASS |
| DOMA dynamic import names | PASS |
| Official YAML semantic parity after DOMA-only normalization | PASS |
| V1/V2/V3 module and RNG isolation | PASS |
| V2/V3 explicit no-context/no-geometry/no-quality/no-refiner ablations | PASS |
| Disabled consensus contributes no consensus loss | PASS |
| Centered-yaw zero-residual identity | PASS |
| V1/V2/V3 CPU forward/loss/backward | PASS |
| V1/V2/V3 same-forward inference contract | PASS |
| Official/DOMA full-model parameter and state counts | PASS |
| Stage2 trainability and merge fail-closed checks | PASS |
| Official critical-file zero diff | PASS |

## 7. Minimum server smoke checklist

No long training is needed for initial acceptance. Run, in order:

1. Re-run the two source-parity gates.
2. Run fixed-checkpoint Official inference parity.
3. Run the fixed-state Official one-step parity capture.
4. Instantiate each DOMA Stage1 YAML on CUDA and print parameter/state counts.
5. For each V1/V2/V3, run one Stage1 training batch and one validation batch.
6. For one modality per version, load Stage1 into Stage2 and run one backward
   step; confirm only the intended active adapters and Official Stage2 modules
   receive gradients.
7. Create synthetic m2/m3/m4/m1 checkpoint copies and run
   `python -m opencood.tools.doma_tools merge_final ...`; then intentionally
   perturb a frozen shared tensor and confirm merge fails.
8. Load each final-inference YAML and merged checkpoint; run one scene through
   the DOMA ordered-inference helper and confirm scores/count/NMS contract.

## 8. Current assessment

Local Legacy Training Parity status: **PASS at the source/static gate**.

The remaining evidence is environmental rather than a known code defect:
full dataset/CUDA inference equality and fixed-state training-step equality
must be executed on the server holding the Official checkpoint and serialized
batch. Windows CPU diagnostics cannot establish CUDA-kernel or dataset-level AP
parity, and no long training was run locally.

An old V2/V3 checkpoint has the same refiner key names and shapes but used the
legacy `sin_cos` yaw semantics. It must not be silently loaded into canonical
DOMA V2/V3, which use `sin_cos_centered`; retrain from the clean profile or
perform an explicitly reviewed semantic conversion. This warning concerns old
DOMA checkpoint compatibility, not Official HEAL parity.
