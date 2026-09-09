"""CPU/static acceptance checks for optional bestdet checkpoint selection."""

import os
import random
import tempfile

import numpy as np
import torch
import torch.nn as nn
import yaml

from opencood.tools import seed_utils, train_utils, validation_detection
from opencood.tools.heal_tools import get_model_path_from_dir


def _enabled_config(**overrides):
    raw = {
        "enabled": True,
        "metric": "ap70",
        "eval_freq": 2,
        "save_bestdet": True,
        "keep_only_best": True,
    }
    raw.update(overrides)
    return validation_detection.get_checkpoint_selection_config(
        {"checkpoint_selection": raw})


def _metrics(ap70, ap30=0.7, ap50=0.6):
    return {"ap30": ap30, "ap50": ap50, "ap70": ap70}


def _expect_error(function, contains=None):
    try:
        function()
    except Exception as exc:
        if contains is not None:
            assert contains in str(exc), str(exc)
        return exc
    raise AssertionError("expected an exception")


def _rng_snapshot():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": ([state.clone() for state in torch.cuda.get_rng_state_all()]
                 if torch.cuda.is_available() else None),
    }


def _assert_rng_equal(before, after):
    assert before["python"] == after["python"]
    assert before["numpy"][0] == after["numpy"][0]
    assert np.array_equal(before["numpy"][1], after["numpy"][1])
    assert before["numpy"][2:] == after["numpy"][2:]
    assert torch.equal(before["torch"], after["torch"])
    if before["cuda"] is not None:
        assert len(before["cuda"]) == len(after["cuda"])
        assert all(torch.equal(left, right)
                   for left, right in zip(before["cuda"], after["cuda"]))


class _RngDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        random.random()
        np.random.random()
        torch.rand(1)
        return index

    @staticmethod
    def collate_batch_test(batch):
        return batch[0]


def _check_rng_guard(seed):
    seed_utils.seed_everything(seed)
    loader = validation_detection.build_detection_validation_loader(
        _RngDataset(), seed, num_workers=0)
    before = _rng_snapshot()
    with validation_detection.preserve_global_rng_state():
        list(loader)
    after = _rng_snapshot()
    _assert_rng_equal(before, after)


def _save_plain_checkpoint(path, value):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    torch.save(model.state_dict(), path)
    return model


def _commit(model, directory, config, run_id, epoch, ap70, state):
    return validation_detection.commit_detection_evaluation(
        model, directory, config, run_id, epoch, _metrics(ap70), *state)


def _state_tuple(result):
    return result[0], result[1], result[2]


def run_checks():
    results = []

    # 1-4: the seed and checkpoint-selection axes are independent.
    assert seed_utils.get_seed({}) is None
    assert not validation_detection.get_checkpoint_selection_config({})["enabled"]
    results.append("01 default official")

    assert seed_utils.get_seed({"seed": 303}) == 303
    assert not validation_detection.get_checkpoint_selection_config(
        {"seed": 303})["enabled"]
    results.append("02 seed only")

    bestdet_only = {"checkpoint_selection": {"enabled": True}}
    assert seed_utils.get_seed(bestdet_only) is None
    assert validation_detection.get_checkpoint_selection_config(
        bestdet_only)["enabled"]
    results.append("03 bestdet only")

    both = {"seed": 303, "checkpoint_selection": {"enabled": True}}
    assert seed_utils.get_seed(both) == 303
    assert validation_detection.get_checkpoint_selection_config(both)["enabled"]
    results.append("04 seed plus bestdet")

    # 5-6: disabled is inert; enabled remains strict.
    disabled = validation_detection.get_checkpoint_selection_config({
        "checkpoint_selection": {
            "enabled": False,
            "metric": "xxx",
            "eval_freq": 0,
            "save_bestdet": "invalid",
        }
    })
    assert disabled == validation_detection.get_checkpoint_selection_config({})
    results.append("05 disabled invalid metric")

    _expect_error(
        lambda: validation_detection.get_checkpoint_selection_config({
            "checkpoint_selection": {"enabled": True, "metric": "xxx"}
        }), "Unsupported checkpoint selection metric")
    results.append("06 enabled invalid metric")

    _check_rng_guard(None)
    results.append("07 RNG guard without seed")
    _check_rng_guard(303)
    results.append("08 RNG guard with seed")

    model = nn.Linear(1, 1)
    config = _enabled_config()

    with tempfile.TemporaryDirectory() as directory:
        state = validation_detection.restore_detection_selection_state(
            directory, config)
        assert state == (-float("inf"), None, [], None)
    results.append("09 fresh bestdet")

    with tempfile.TemporaryDirectory() as directory:
        state = (-float("inf"), None, [])
        committed = _commit(model, directory, config, "run-a", 18, 0.5, state)
        restored = validation_detection.restore_detection_selection_state(
            directory, config)
        assert restored[:3] == committed[:3]
        assert restored[3] == "net_epoch_bestdet_at18.pth"
    results.append("10 valid resume")

    with tempfile.TemporaryDirectory() as directory:
        validation_detection.save_bestdet_checkpoint(model, directory, 18)
        _expect_error(
            lambda: validation_detection.restore_detection_selection_state(
                directory, config),
            "Existing bestdet checkpoint found")
    results.append("11 missing history fails fast")

    with tempfile.TemporaryDirectory() as directory:
        history_path = os.path.join(
            directory, validation_detection.HISTORY_FILENAME)
        with open(history_path, "w", encoding="utf-8") as stream:
            stream.write("history: [unterminated")
        _expect_error(
            lambda: validation_detection.restore_detection_selection_state(
                directory, config), "cannot be read")
    results.append("12 corrupt history fails fast")

    with tempfile.TemporaryDirectory() as directory:
        state = _state_tuple(
            _commit(model, directory, config, "run-a", 18, 0.5,
                    (-float("inf"), None, [])))
        assert state[1] == 18
        _expect_error(
            lambda: validation_detection.restore_detection_selection_state(
                directory, _enabled_config(metric="ap50")),
            "Restore the original metric")
    results.append("13 metric mismatch fails fast")

    with tempfile.TemporaryDirectory() as directory:
        _commit(model, directory, config, "run-a", 18, 0.5,
                (-float("inf"), None, []))
        os.replace(
            os.path.join(directory, "net_epoch_bestdet_at18.pth"),
            os.path.join(directory, "net_epoch_bestdet_at20.pth"))
        _expect_error(
            lambda: validation_detection.restore_detection_selection_state(
                directory, config), "not committed in history")
    results.append("14 mismatched checkpoint fails fast")

    with tempfile.TemporaryDirectory() as directory:
        history = [{"run_id": "run-a", "epoch": 2, **_metrics(0.4)}]
        for _ in range(2):
            validation_detection.save_detection_history(
                directory, _enabled_config(save_bestdet=False), 2, 0.4,
                history, None)
        history_path = os.path.join(
            directory, validation_detection.HISTORY_FILENAME)
        with open(history_path, encoding="utf-8") as stream:
            assert yaml.safe_load(stream)["best_epoch"] == 2
        assert not os.path.exists(history_path + ".tmp")
        restored = validation_detection.restore_detection_selection_state(
            directory, _enabled_config(save_bestdet=False))
        assert restored[:3] == (0.4, 2, history)
        assert restored[3] is None
    results.append("15 atomic history")

    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = validation_detection.save_bestdet_checkpoint(
            model, directory, 3)
        assert os.path.exists(checkpoint_path)
        assert not os.path.exists(checkpoint_path + ".tmp")
        assert isinstance(torch.load(checkpoint_path, map_location="cpu"), dict)
    results.append("16 atomic checkpoint")

    with tempfile.TemporaryDirectory() as directory:
        first = _commit(model, directory, config, "run-a", 18, 0.5,
                        (-float("inf"), None, []))
        original_dump = validation_detection._atomic_yaml_dump

        def fail_history(_data, _path):
            raise OSError("simulated history failure")

        validation_detection._atomic_yaml_dump = fail_history
        try:
            _expect_error(
                lambda: _commit(model, directory, config, "run-a", 21, 0.6,
                                _state_tuple(first)),
                "simulated history failure")
        finally:
            validation_detection._atomic_yaml_dump = original_dump
        assert os.path.exists(
            os.path.join(directory, "net_epoch_bestdet_at18.pth"))
        assert os.path.exists(
            os.path.join(directory, "net_epoch_bestdet_at21.pth"))
        _expect_error(
            lambda: validation_detection.restore_detection_selection_state(
                directory, config), "not committed in history")
    results.append("17 new-best transaction order")

    with tempfile.TemporaryDirectory() as directory:
        first = _commit(model, directory, config, "run-a", 18, 0.5,
                        (-float("inf"), None, []))
        second = _commit(model, directory, config, "run-a", 21, 0.6,
                         _state_tuple(first))
        _save_plain_checkpoint(
            os.path.join(directory, "net_epoch_bestdet_at18.pth"), 18)
        restored = validation_detection.restore_detection_selection_state(
            directory, config)
        assert restored[1] == second[1] == 21
        assert not os.path.exists(
            os.path.join(directory, "net_epoch_bestdet_at18.pth"))
        assert os.path.exists(
            os.path.join(directory, "net_epoch_bestdet_at21.pth"))
    results.append("18 stale checkpoint recovery")

    with tempfile.TemporaryDirectory() as directory:
        first = _commit(model, directory, config, "first-run", 18, 0.5,
                        (-float("inf"), None, []))
        second = _commit(model, directory, config, "second-run", 18, 0.6,
                         _state_tuple(first))
        restored = validation_detection.restore_detection_selection_state(
            directory, config)
        assert restored[1] == 18
        assert [item["run_id"] for item in restored[2]] == [
            "first-run", "second-run"]
        assert second[4]
    results.append("19 run_id repeat epoch")

    with tempfile.TemporaryDirectory() as directory:
        with open(os.path.join(directory, "config.yaml"), "w",
                  encoding="utf-8") as stream:
            yaml.safe_dump({"checkpoint_selection": {"enabled": False}}, stream)
        _save_plain_checkpoint(
            os.path.join(directory, "net_epoch_bestval_at15.pth"), 15)
        _save_plain_checkpoint(
            os.path.join(directory, "net_epoch_bestdet_at21.pth"), 21)
        _save_plain_checkpoint(
            os.path.join(directory, "net_epoch25.pth"), 25)
        loaded = nn.Linear(1, 1, bias=False)
        epoch, loaded = train_utils.load_saved_model(directory, loaded)
        assert epoch == 15 and loaded.weight.item() == 15
        assert os.path.basename(get_model_path_from_dir(directory)) == \
            "net_epoch_bestval_at15.pth"
        os.remove(os.path.join(directory, "net_epoch_bestval_at15.pth"))
        epoch, loaded = train_utils.load_saved_model(directory, loaded)
        assert epoch == 25 and loaded.weight.item() == 25
        assert os.path.basename(get_model_path_from_dir(directory)) == \
            "net_epoch25.pth"

    with tempfile.TemporaryDirectory() as directory:
        with open(os.path.join(directory, "config.yaml"), "w",
                  encoding="utf-8") as stream:
            yaml.safe_dump({"checkpoint_selection": {
                "enabled": True, "metric": "ap70", "eval_freq": 2,
                "save_bestdet": True, "keep_only_best": True,
            }}, stream)
        _commit(model, directory, config, "run-a", 21, 0.6,
                (-float("inf"), None, []))
        selected = validation_detection.resolve_selected_checkpoint_from_model_dir(
            directory)
        assert os.path.basename(selected) == "net_epoch_bestdet_at21.pth"
        assert get_model_path_from_dir(directory) == selected
        epoch, _ = train_utils.load_saved_model(
            directory, nn.Linear(1, 1), checkpoint_path=selected)
        assert epoch == 21
    results.append("20 checkpoint loader isolation")

    print("checkpoint_selection checks: PASS")
    for result in results:
        print("PASS " + result)
    return results


if __name__ == "__main__":
    run_checks()
