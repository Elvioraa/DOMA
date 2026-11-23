"""CPU regression tests and explicit server-only preflight probes."""

import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile

import torch
import yaml

from opencood.tools.table4_ext import (ROOT, ROWS, STAGES, CONFIGS, config,
                                      parse_ap, plan, require)


def switches():
    from opencood.tools.check_doma_functional import _Holder
    from opencood.models.sub_modules.doma_object import (
        configure_doma_trainability, predict_scene_residuals, run_doma_training)
    from opencood.loss.doma_object_loss import compute_doma_object_loss
    from opencood.tools.doma_tools import apply_doma_merge_ownership, SHARED_PREFIXES
    for row, flags in ROWS.items():
        holders = {}
        for stage in STAGES:
            args = yaml.safe_load(config(row, stage).read_text())["model"]["args"]
            model = _Holder(args)
            model._doma_log_printed = True
            configure_doma_trainability(model)
            holders[stage] = model
            names = [name for name, p in model.named_parameters() if p.requires_grad]
            require(bool(names) == (stage != "final_infer"), "Trainability mismatch")
            if stage.startswith("stage2/"):
                modality = stage.split("/")[1]
                require(all(name.startswith(("doma_object_adapter_" + modality + ".",
                                             "doma_context_adapter_" + modality + "."))
                            for name in names), "Stage2 shared tensors must be frozen")
            require(hasattr(model, "doma_shared_context_encoder") == bool(flags[0]), "M independence")
            require(hasattr(model, "doma_shared_geometry_encoder") == bool(flags[1]), "G independence")
            require(model.doma_shared_object_refiner.network[0].in_features == 160,
                    "G off must preserve refiner width")
        model = holders["stage1"]
        features = torch.randn(2, 64, 32, 32, requires_grad=True)
        scene = dict(agent_features=features, agent_support=torch.ones(2, 1, 32, 32),
                     agent_modalities=("m1", "m1"))
        if flags[0]:
            scene.update(context_agent_features=torch.randn(2, 128, 16, 16, requires_grad=True),
                         context_agent_support=torch.ones(2, 1, 16, 16))
        data = dict(object_bbx_center=torch.tensor([[[0., 0., -1., 1.56, 1.6, 3.9, .2]]]),
                    object_bbx_mask=torch.ones(1, 1, dtype=torch.bool))
        payload = run_doma_training(model, dict(scenes=(scene,), box_order="hwl", aligned_to="ego"), data)
        loss, stats = compute_doma_object_loss(payload)
        require(bool(torch.isfinite(loss)), "Nonfinite loss")
        loss.backward()
        require(features.grad is not None, "M=0 lost detail gradient")
        require(payload["consensus_enabled"] == bool(flags[2]), "C flag mismatch")
        if not flags[2]:
            require(stats["doma_consensus_loss"] == 0, "C=0 consensus loss active")
        # Give each member distinguishable predictions; verify mean vs ego rule.
        with torch.no_grad():
            torch.nn.init.normal_(model.doma_shared_object_refiner.network[-1].weight, std=.05)
            result = predict_scene_residuals(model, scene, data["object_bbx_center"][0])
            individual = result["per_agent_residuals"]
            valid = result["valid_mask"]
            expected = ((individual * valid.unsqueeze(-1)).sum(1) /
                        valid.sum(1, keepdim=True).clamp_min(1)) if flags[2] else individual[:, 0]
            torch.testing.assert_close(result["fused_residuals"], expected)
        anchor = holders["stage1"].state_dict()
        states = []
        for modality in ("m2", "m3", "m4"):
            state = copy.deepcopy(holders["stage2/" + modality].state_dict())
            for key, value in anchor.items():
                if key.startswith(SHARED_PREFIXES):
                    state[key] = value.clone()
            states.append(state)
        states.append(anchor)
        merged = apply_doma_merge_ownership({}, states)
        holders["final_infer"].load_state_dict(merged, strict=True)
        print(row, "PASS forward/backward, M/G/C semantics, trainability, strict object merge")


def state_roundtrip():
    from opencood.tools import table4_ext_state as state
    class Dataset:
        def reinitialize(self):
            import random
            self.order = list(range(10))
            random.shuffle(self.order)
    class Loader:
        generator = None
    with tempfile.TemporaryDirectory(prefix="table4-ext-test-") as directory:
        root = Path(directory)
        (root / "config.yaml").write_text("seed: 303\n")
        torch.manual_seed(303)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2], gamma=.1)
        data = torch.ones(3, 2)
        model(data).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        state.atomic_save(model.state_dict(), root / "net_epoch_bestval_at1.pth")
        loaders = [Loader(), Loader()]
        for loader in loaders:
            loader.generator = torch.Generator().manual_seed(303)
        dataset = Dataset()
        before = state.rng_state()
        dataset.reinitialize()
        order = list(dataset.order)
        state.save_epoch(root, model, optimizer, scheduler, None, 1, .2, 1, loaders, before, 1)
        model(data).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        expected = copy.deepcopy(model.state_dict())
        restored = state.restore_epoch(root, model, optimizer, scheduler, None, loaders, dataset)
        require(restored["next_epoch"] == 1 and dataset.order == order, "Dataset/epoch recovery failed")
        model(data).sum().backward()
        optimizer.step()
        for key, value in expected.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        (root / "config.yaml").write_text("seed: 999\n")
        try:
            state.restore_epoch(root, model, optimizer, scheduler, None, loaders, dataset)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Changed config accepted")
    print("PASS Adam/scheduler/RNG/dataset epoch recovery; config drift rejected")


def controller_tests():
    """Exercise real orchestration decisions with fake training processes, on CPU."""
    from unittest.mock import patch
    import shutil
    import sys
    import types
    from opencood.tools import table4_ext as runner
    from opencood.tools.table4_ext_state import atomic_save

    class Process:
        fail = False
        calls = []

        def __init__(self, command, cwd, stdout, stderr):
            self.pid = 12345
            self.code = 17 if Process.fail else 0
            Process.calls.append(command)
            if self.code:
                stdout.write("simulated failure\n")
                return
            if "merge_final" in command:
                atomic_save({"weight": torch.tensor([1.])}, Path(command[-1]) / "net_epoch1.pth")
                stdout.write("simulated merge\n")
                return
            directory = Path(command[command.index("--model_dir") + 1])
            if "opencood.tools.inference_doma_heter_in_order" in command:
                stdout.write(("The Average Precision at IOU 0.3 is 0.9, "
                              "The Average Precision at IOU 0.5 is 0.8, "
                              "The Average Precision at IOU 0.7 is 0.7\n") * 4)
                return
            cfg = yaml.safe_load((directory / "config.yaml").read_text())
            atomic_save({"weight": torch.tensor([1.])}, directory / "net_epoch_bestval_at3.pth")
            atomic_save(dict(next_epoch=cfg["train_params"]["epoches"],
                             best_epoch=3, best_loss=.1), directory / "resume_state.pth")
            stdout.write("Training Finished, checkpoints saved\n")

        def wait(self):
            return self.code

    with tempfile.TemporaryDirectory(prefix="table4-controller-") as directory:
        root = Path(directory)
        with patch.object(runner, "OUTPUT", root), patch.object(runner, "gpu_idle", return_value={}):
            for row in ROWS:
                for stage in STAGES:
                    dest = root / row / stage / "config.yaml"
                    dest.parent.mkdir(parents=True)
                    shutil.copy2(config(row, stage), dest)
                manifest = dict(experiment=row, status="PENDING", stages={})
                with patch.object(runner.subprocess, "Popen", Process):
                    for stage in STAGES[:-1]:
                        runner.train_stage(row, stage, manifest)
                    count = len(Process.calls)
                    for stage in STAGES[:-1]:
                        runner.train_stage(row, stage, manifest)
                    require(len(Process.calls) == count, "Completed stage was rerun")
            # Corruption must not trigger a silent skip or a destructive retrain.
            selected = root / row / "stage1/net_epoch_bestval_at3.pth"
            selected.write_bytes(b"corrupt")
            try:
                runner.valid_completion(row, "stage1", manifest["stages"]["stage1"])
            except RuntimeError:
                pass
            else:
                raise AssertionError("Corrupt checkpoint accepted")
            # Nonzero subprocess exit cannot mark a stage COMPLETE.
            Process.fail = True
            manifest = dict(experiment="G_C", status="PENDING", stages={})
            with patch.object(runner.subprocess, "Popen", Process):
                try:
                    runner.execute(["fake"], root / "failure", manifest, "stage1")
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("Failed training accepted")
            record = manifest['stages']['stage1']
            require(record['status'] != 'COMPLETE' and record['attempts'][-1]['exit_code'] == 17,
                    "Failure lost its exit code")
            runner.summarize()
            text = (root / "ablation_results.csv").read_text()
            require(all(name in text for name in ROWS), "Missing combinations in summary")
    # Simulate the whole scheduler including merge, inference and crash handling.
    # The Linux flock itself is NOT validated by this platform-independent test.
    fake_fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
    for fail in (False, True):
        with tempfile.TemporaryDirectory(prefix="table4-workflow-") as directory:
            root = Path(directory)
            Process.fail = fail
            Process.calls = []
            with patch.object(runner, "OUTPUT", root), \
                 patch.object(runner, "gpu_idle", return_value={}), \
                 patch.object(runner, "boot_id", return_value="test"), \
                 patch.object(runner, "data_fingerprint", return_value={}), \
                 patch.object(runner, "code_fingerprint", return_value={}), \
                 patch.object(runner, "capture", return_value="test"), \
                 patch.object(runner, "run", return_value=None), \
                 patch.object(runner.subprocess, "Popen", Process), \
                 patch.dict(sys.modules, {"fcntl": fake_fcntl}):
                if fail:
                    try:
                        runner.worker()
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError("Worker ignored training failure")
                    require(runner.read_json(root / "G_C/run_manifest.json")["status"] == "FAILED",
                            "Row failure not persisted")
                    require(len(Process.calls) == 1, "Dependent stages ran after failure")
                    require(runner.read_json(root / "M_C/run_manifest.json")["status"] == "PENDING",
                            "Later row incorrectly advanced")
                else:
                    runner.worker()
                    require(len(Process.calls) == 24, "Wrong pipeline stage count/order")
                    for row in ROWS:
                        manifest = runner.read_json(root / row / "run_manifest.json")
                        require(manifest["status"] == "COMPLETE", "Row not completed")
                        require(manifest["stages"]["final_infer"]["ap"] == [.9, .8, .7], "Wrong AP")
                    runner.worker()
                    require(len(Process.calls) == 24, "Verified completed workflow reran")
    print("PASS controller: 16 stages, skip, corrupt checkpoint rejection, failed exit, all-row summary")
    print("PASS simulated full workflow: four rows train/merge/infer; failure stops dependents")


def server_cpu():
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils
    # Actual imports, never import stubs or fallback dependency substitutes.
    from opencood.data_utils.datasets import build_dataset
    import opencood.tools.inference_doma_heter_in_order
    import opencood.tools.train_table4_ext
    for row in ROWS:
        for stage in STAGES:
            cfg = yaml_utils.load_yaml(str(config(row, stage)))
            for key in ("root_dir", "validate_dir", "test_dir"):
                require((ROOT / cfg[key]).is_dir(), "Missing data directory " + cfg[key])
            require((ROOT / cfg["heter"]["assignment_path"]).is_file(), "Missing assignment")
            model = train_utils.create_model(cfg)
            if hasattr(model, "model_train_init") and stage != "final_infer":
                model.model_train_init()
            trainable = [name for name, p in model.named_parameters() if p.requires_grad]
            require(bool(trainable), "No model parameters available")
            if stage.startswith("stage2/"):
                require(not any(name.startswith("doma_shared_") for name in trainable),
                        "Stage2 shared weights trainable")
            del model
    # Build each modality's real dataset once; checks assignments and symlinks.
    for stage in STAGES[:-1]:
        cfg = yaml_utils.load_yaml(str(config("G_C", stage)))
        for train in (True, False):
            dataset = build_dataset(cfg, visualize=False, train=train)
            require(len(dataset) > 0, "Empty dataset")
            del dataset
    switches()
    state_roundtrip()
    print("SERVER CPU dependencies/model/dataset construction PASS")


def ddp():
    import torch.distributed as dist
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0,2", "Wrong GPU mask")
    local = int(os.environ["LOCAL_RANK"])
    require(torch.cuda.device_count() == 2, "Exactly two logical GPUs expected")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        # Compare CUDA UUID with the selected nvidia-smi UUID, not device names.
        import ctypes
        from opencood.tools.table4_ext import capture
        class UUID(ctypes.Structure):
            _fields_ = [("value", ctypes.c_ubyte * 16)]
        cuda = ctypes.CDLL("libcuda.so.1")
        require(cuda.cuInit(0) == 0, "cuInit failed")
        identifier = UUID()
        require(cuda.cuDeviceGetUuid(ctypes.byref(identifier), local) == 0, "CUDA UUID query failed")
        raw = bytes(identifier.value).hex()
        actual = "GPU-" + "-".join((raw[:8], raw[8:12], raw[12:16], raw[16:20], raw[20:]))
        physical = (0, 2)[local]
        expected = capture(["nvidia-smi", "-i", str(physical), "--query-gpu=uuid", "--format=csv,noheader"])
        require(actual == expected, "Logical/physical GPU UUID mismatch")
        value = torch.tensor([local + 1.], device="cuda")
        dist.all_reduce(value)
        require(value.item() == 3, "NCCL all-reduce failed")
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(4, 1).cuda(), device_ids=[local])
        model(torch.ones(2, 4, device="cuda")).sum().backward()
        print("DDP rank", local, "physical", physical, "uuid", actual, "PASS", flush=True)
    finally:
        dist.destroy_process_group()


def merged(directory):
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils
    from opencood.tools.table4_ext import checkpoint
    cfg = yaml_utils.load_yaml(str(Path(directory) / "config.yaml"))
    model = train_utils.create_model(cfg)
    model.load_state_dict(checkpoint(Path(directory) / "net_epoch1.pth"), strict=True)
    print("Full merged model strict checkpoint load PASS")


def batch():
    """Server-only real DDP batch forward/backward; no optimizer or checkpoint."""
    import gc
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils, seed_utils
    from opencood.data_utils.datasets import build_dataset
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    try:
        for row in ROWS:
            for stage in STAGES[:-1]:
                cfg = yaml_utils.load_yaml(str(config(row, stage)))
                seed_utils.seed_from_hypes(cfg)
                dataset = build_dataset(cfg, visualize=False, train=True)
                loader = DataLoader(dataset, batch_size=cfg["train_params"]["batch_size"],
                                    num_workers=0, collate_fn=dataset.collate_batch_train,
                                    **seed_utils.dataloader_seed_kwargs(303, local))
                batch_data = next(iter(loader))
                require(batch_data is not None, "First real batch empty")
                batch_data = train_utils.to_device(batch_data, torch.device("cuda", local))
                batch_data['ego']['epoch'] = 0
                model = train_utils.create_model(cfg).cuda()
                model.train()
                if hasattr(model, "model_train_init"):
                    model.model_train_init()
                ddp_model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[local], find_unused_parameters=True)
                loss_fn = train_utils.create_loss(cfg)
                output = ddp_model(batch_data['ego'])
                loss = loss_fn(output, batch_data['ego']['label_dict'])
                require(bool(torch.isfinite(loss)), "Nonfinite batch loss")
                loss.backward()
                print("Real DDP batch PASS", row, stage, "rank", local, flush=True)
                del loss, output, ddp_model, model, loss_fn, batch_data, loader, dataset
                gc.collect()
                torch.cuda.empty_cache()
                dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-cpu", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--merged")
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.batch:
        batch()
    elif args.ddp:
        ddp()
    elif args.merged:
        merged(args.merged)
    elif args.server_cpu:
        server_cpu()
    else:
        plan()
        switches()
        state_roundtrip()
        controller_tests()
        sample = ("The Average Precision at IOU 0.3 is 0.9, The Average Precision at IOU 0.5 is 0.8, "
                  "The Average Precision at IOU 0.7 is 0.7\n")
        require(parse_ap(sample * 4) == (.9, .8, .7), "AP parsing failed")
        for bad in (sample, sample * 5, (sample * 4).replace("0.9,", "1.9,")):
            try:
                parse_ap(bad)
            except RuntimeError:
                continue
            raise AssertionError("Malformed AP accepted")
        print("LOCAL EXT TESTS PASS; no real CUDA/DDP or dataset forward tested")


if __name__ == "__main__":
    main()
