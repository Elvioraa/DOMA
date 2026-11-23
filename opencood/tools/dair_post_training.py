"""Opt-in, current-stage DAIR bestval AP; never selects or merges Final weights."""
import copy
import math
import tempfile
from pathlib import Path

from opencood.tools.dair_protocol import (
    digest, enabled, load_state, load_training_state, require, validate_protocol,
)


def is_enabled(cfg):
    raw = cfg.get('train_params', {}).get('dair_bestval_ap', {})
    if not isinstance(raw, dict) or type(raw.get('enabled', False)) is not bool:
        raise ValueError('train_params.dair_bestval_ap.enabled must be bool')
    if not raw.get('enabled', False):
        return False
    require(enabled(cfg), 'dair_bestval_ap requires the explicit DAIR protocol')
    validate_protocol(cfg)
    require(cfg['dair_protocol']['stage'] in ('stage1', 'stage2'), 'Training stage required')
    require(not cfg.get('checkpoint_selection', {}).get('enabled', False),
            'dair_bestval_ap requires unchanged Validation Loss checkpoint selection')
    return True


class _CountedLoader:
    def __init__(self, loader):
        self.loader = loader
        self.processed = 0
        self.skipped = 0

    def __iter__(self):
        for batch in self.loader:
            if batch is None:
                self.skipped += 1
            else:
                self.processed += 1
            yield batch


def run_bestval_ap(cfg, directory, best_epoch, model, dataset, device):
    """Return False when disabled; failures propagate after a separate failure record.

    The caller supplies its exact Validation Loss winner and unwrapped model.
    Evaluation reuses the native train=False validation dataset and existing AP helper.
    Each invocation creates a new directory; previous records are never overwritten.
    """
    if not is_enabled(cfg):
        return False
    from opencood.tools import validation_detection, seed_utils
    from opencood.tools.inference_dair import _report_json

    directory = Path(directory).resolve()
    output = Path(tempfile.mkdtemp(prefix='dair_bestval_ap_', dir=str(directory)))
    report = dict(status='running', evaluation='current_stage_validation_not_final_merge',
                  stage=cfg['dair_protocol']['stage'], selection='validation_loss',
                  best_epoch=best_epoch, effective_config=copy.deepcopy(cfg))
    counted = None
    try:
        require(type(best_epoch) is int and best_epoch > 0, 'No Validation Loss best epoch')
        checkpoint = directory / ('net_epoch_bestval_at%d.pth' % best_epoch)
        require(checkpoint.is_file(), 'Selected bestval checkpoint is missing: ' + str(checkpoint))
        report.update(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                      validation_path=str(Path(cfg['validate_dir']).resolve()),
                      validation_sha256=digest(cfg['validate_dir']),
                      expected_samples=len(dataset))
        require(len(dataset) == cfg['dair_protocol']['validation_samples'], 'Unexpected Validation size')
        fusion = cfg['fusion']['core_method'].removesuffix('heter')
        validation_detection.validate_fusion_method(fusion)
        report['fusion_method'] = fusion
        # Snapshot before model loading/evaluation; contains the parsed anchor grid.
        (output / 'effective_protocol.json').write_text(_report_json(report), encoding='utf8')
        load_training_state(model, load_state(checkpoint), directory, checkpoint, cfg)
        loader = validation_detection.build_detection_validation_loader(
            dataset, seed_utils.get_seed(cfg))
        counted = _CountedLoader(loader)
        print('Evaluating current-stage DAIR bestval AP: %s' % checkpoint, flush=True)
        metrics = validation_detection.evaluate_detection_ap(model, counted, dataset, device, fusion)
        require(counted.processed == len(dataset) and counted.skipped == 0,
                'Incomplete Validation evaluation; cannot report full-split AP')
        require(all(math.isfinite(float(metrics[k])) for k in ('ap30', 'ap50', 'ap70')),
                'Non-finite detection AP')
        require(digest(checkpoint) == report['checkpoint_sha256'], 'Checkpoint changed during evaluation')
        report.update(status='success', metrics=metrics, evaluated_samples=counted.processed,
                      skipped_samples=counted.skipped)
        (output / 'result.json').write_text(_report_json(report), encoding='utf8')
        print('The Average Precision at IOU 0.3 is %.4f, '
              'The Average Precision at IOU 0.5 is %.4f, '
              'The Average Precision at IOU 0.7 is %.4f' %
              (metrics['ap30'], metrics['ap50'], metrics['ap70']), flush=True)
        print('DAIR bestval AP record: %s' % (output / 'result.json'), flush=True)
    except Exception as exc:
        report.update(status='failed', error=str(exc),
                      evaluated_samples=0 if counted is None else counted.processed,
                      skipped_samples=0 if counted is None else counted.skipped)
        (output / 'failure.json').write_text(_report_json(report), encoding='utf8')
        raise RuntimeError('Training completed, but DAIR bestval AP FAILED; record: %s' % output) from exc
    return True
