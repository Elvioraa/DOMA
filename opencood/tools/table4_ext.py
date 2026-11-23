"""Sequential, fail-closed Table 4 experiments. No training on import/check/status."""

import argparse
import ast
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "opencood/hypes_yaml/opv2v/MoreModality/DOMA/V2_R2_TABLE4_EXT"
OUTPUT = ROOT / "opencood/logs/DOMA_V2_R2_2GPU_TABLE4_EXT_ABLATION_SEED303"
ROWS = {"G_C": (0, 1, 1), "M_C": (1, 0, 1),
        "G_Only": (0, 1, 0), "C_Only": (0, 0, 1)}
STAGES = ("stage1", "stage2/m2", "stage2/m3", "stage2/m4", "final_infer")
HISTORY = "opencood/logs/DOMA_V2_R2_2GPU_TABLE4_ABLATION_SEED303/"
FULL = "opencood/logs/DOMA_V2_R2_2GPU_LR1P414_SEED303/"
HIST_STAGES = ("stage1/m1_base", "stage2/m2_alignto_m1_lr0p0012",
               "stage2/m3_alignto_m1_lr0p0030", "stage2/m4_alignto_m1_lr0p0017",
               "final_infer")
AP_PATTERN = re.compile(
    r"The Average Precision at IOU 0\.3 is ([0-9.]+), "
    r"The Average Precision at IOU 0\.5 is ([0-9.]+), "
    r"The Average Precision at IOU 0\.7 is ([0-9.]+)")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with Path(source).open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(str(temporary), str(destination))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def run(args, **kwargs):
    return subprocess.run([str(a) for a in args], cwd=str(ROOT), check=True,
                          text=True, **kwargs)


def capture(args):
    return run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def config(row, stage):
    return CONFIGS / row / stage / "config.yaml"


def parse_ap(text):
    matches = AP_PATTERN.findall(text)
    require(len(matches) == 4, "Expected exactly four ordered CAV1-CAV4 AP records")
    values = [tuple(float(x) for x in row) for row in matches]
    require(all(0 <= x <= 1 for row in values for x in row), "Invalid AP value")
    return values[-1]


def code_fingerprint():
    paths = sorted(p for p in (ROOT / "opencood").rglob("*.py")
                   if "logs" not in p.relative_to(ROOT).parts)
    paths += sorted(CONFIGS.rglob("*config.yaml"))
    paths += [CONFIGS / "history_snapshot.json", ROOT / "scripts/run_doma_table4_ext_ablation.sh"]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def data_fingerprint():
    """Path/split inventory and assignment digest, not a hash of all sensor bytes."""
    result = {}
    for stage in STAGES:
        cfg = yaml.safe_load(config("G_C", stage).read_text(encoding="utf8"))
        for key in ("root_dir", "validate_dir", "test_dir"):
            path = ROOT / cfg[key]
            require(path.is_dir(), "Missing dataset/symlink: " + str(path))
            result[cfg[key]] = dict(resolved=str(path.resolve()),
                                    scenarios=sorted(p.name for p in path.iterdir() if p.is_dir()))
        assignment = ROOT / cfg["heter"]["assignment_path"]
        result[cfg["heter"]["assignment_path"]] = dict(resolved=str(assignment.resolve()),
                                                       sha256=digest(assignment))
    return result


def environment_fingerprint():
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for package in ("torch", "torchvision", "numpy", "PyYAML", "spconv", "spconv-cu117",
                    "timm", "tensorboardX", "efficientnet-pytorch", "open3d"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()), packages=versions)


def plan():
    from opencood.models.sub_modules.doma_config import validate_doma_config
    snapshot = read_json(CONFIGS / "history_snapshot.json")
    result = {}
    for row, flags in ROWS.items():
        result[row] = {}
        for stage, historical_stage in zip(STAGES, HIST_STAGES):
            path = config(row, stage)
            cfg = yaml.safe_load(path.read_text(encoding="utf8"))
            doma = cfg["model"]["args"]["doma"]
            validate_doma_config(doma)
            require(tuple(int(doma[key]["enabled"]) for key in
                          ("multi_granularity", "geometry", "consensus")) == flags,
                    "Wrong M/G/C in " + str(path))
            require(doma["object_encoder"]["enabled"] and doma["refiner"]["enabled"],
                    "Object branch must remain enabled")
            require(doma["consensus"]["mode"] == "uniform_geometry_mean", "Wrong consensus")
            source = yaml.safe_load(snapshot[HISTORY + "Row2_DualLayer/" +
                                              historical_stage + "/config.yaml"])
            expected = copy.deepcopy(source)
            expected["name"] = cfg["name"]
            d = expected["model"]["args"]["doma"]
            d["ablation"] = True
            if not flags[0]:
                d["multi_granularity"] = {"enabled": False}
            d["geometry"]["enabled"] = bool(flags[1])
            d["consensus"]["enabled"] = bool(flags[2])
            require(cfg == expected, "Protocol drift from frozen Row2 baseline: " + str(path))
            require(cfg["seed"] == 303, "Seed must be 303")
            result[row][stage] = dict(path=str(path), sha256=digest(path),
                                      train=cfg.get("train_params"), optimizer=cfg.get("optimizer"),
                                      scheduler=cfg.get("lr_scheduler"), flags=flags)
    return result


def gpu_idle():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0,2", "CUDA_VISIBLE_DEVICES must be 0,2")
    rows = list(csv.reader(io.StringIO(capture([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total",
        "--format=csv,noheader,nounits"]))))
    selected = {int(row[0]): row for row in rows if int(row[0]) in (0, 2)}
    require(set(selected) == {0, 2}, "Physical GPU 0 and 2 are required")
    uuids = {row[1].strip() for row in selected.values()}
    processes = capture(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                         "--format=csv,noheader"])
    busy = [line for line in processes.splitlines() if line.split(",")[0].strip() in uuids]
    require(not busy, "GPU occupied; no processes killed: " + "; ".join(busy))
    require(all(int(row[3]) < 512 for row in selected.values()), "Unexplained GPU memory use >=512 MiB")
    print("Physical GPU 0,2 -> logical CUDA 0,1:", selected, flush=True)
    return selected


def check(local=False):
    p = plan()
    for file in (ROOT / "opencood/tools").glob("*table4_ext*.py"):
        ast.parse(file.read_text(encoding="utf8"), filename=str(file))
    for row, stages in p.items():
        print(row, ROWS[row], [(s, v["train"]["epoches"], v["train"]["batch_size"])
                              for s, v in stages.items() if s != "final_infer"])
    if local:
        print("LOCAL CONFIG/SYNTAX PASS; CUDA/DDP/data/dependencies NOT VERIFIED")
        return
    require(sys.platform == "linux", "Server preflight requires Linux; use check --local on Windows")
    for binary in ("tmux", "flock", "nvidia-smi", "bash", "git"):
        require(shutil.which(binary), "Missing executable: " + binary)
    run(["bash", "-n", ROOT / "scripts/run_doma_table4_ext_ablation.sh"])
    require(Path(sys.prefix).name == "DOMA", "Use the historical DOMA Conda environment")
    print("Python:", sys.executable, "Conda prefix:", sys.prefix)
    require(shutil.disk_usage(ROOT).free >= 100 * 1024 ** 3, "Require >=100 GiB free disk")
    if (OUTPUT / "execution_manifest.json").exists():
        require(read_json(OUTPUT / "execution_manifest.json")["code_sha256"] == code_fingerprint(),
                "Code/config drift since preparation; refusing to resume")
        require(read_json(OUTPUT / "execution_manifest.json")["data_fingerprint"] == data_fingerprint(),
                "Data paths/splits/assignment changed")
        require(read_json(OUTPUT / "execution_manifest.json")["environment"] == environment_fingerprint(),
                "Python/dependency versions changed")
    # CPU checks use the actual dependency environment; no CUDA stubs.
    run([sys.executable, "-B", "-m", "opencood.tools.check_table4_ext", "--server-cpu"], timeout=600)
    gpu_idle()  # Never initialize CUDA on an occupied selected device.
    run([sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", "--module", "opencood.tools.check_table4_ext", "--ddp"], timeout=180)
    run([sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
         "--nproc_per_node=2", "--module", "opencood.tools.check_table4_ext", "--batch"], timeout=1800)
    print("PREFLIGHT PASS (short CUDA/DDP probe; no training performed)")


def new_manifest(row):
    return dict(experiment=row, switches=dict(zip(("M", "G", "C"), ROWS[row])),
                status="PENDING", seed=303, physical_gpu=[0, 2], logical_gpu=[0, 1],
                stages={}, git_commit=capture(["git", "rev-parse", "HEAD"]),
                code_sha256=code_fingerprint(), created=time.time())


def prepare():
    plan()
    manifest_path = OUTPUT / "execution_manifest.json"
    hashes = code_fingerprint()
    if manifest_path.exists():
        require(read_json(manifest_path)["code_sha256"] == hashes, "Frozen code/config drift")
        require(read_json(manifest_path)["data_fingerprint"] == data_fingerprint(), "Frozen data drift")
        require(read_json(manifest_path)["environment"] == environment_fingerprint(), "Frozen environment drift")
    else:
        write_json(manifest_path, dict(code_sha256=hashes, python=sys.executable,
                   git_commit=capture(["git", "rev-parse", "HEAD"]),
                   git_status=capture(["git", "status", "--short"]),
                   data_fingerprint=data_fingerprint(), environment=environment_fingerprint(), created=time.time()))
    for row in ROWS:
        for stage in STAGES:
            source = config(row, stage)
            destination = OUTPUT / row / stage / "config.yaml"
            if destination.exists():
                require(digest(source) == digest(destination), "Existing config changed: " + str(destination))
            else:
                atomic_copy(source, destination)
        path = OUTPUT / row / "run_manifest.json"
        if not path.exists():
            write_json(path, new_manifest(row))
    summarize()


def checkpoint(path):
    from opencood.tools.table4_ext_state import load
    import torch
    state = load(path)
    require(isinstance(state, dict) and state, "Empty checkpoint: " + str(path))
    require(all(isinstance(k, str) and torch.is_tensor(v) and
                bool(torch.isfinite(v).all()) for k, v in state.items()),
            "Checkpoint contains invalid tensors: " + str(path))
    return state


def best(directory):
    files = list(Path(directory).glob("net_epoch_bestval_at*.pth"))
    require(len(files) == 1, "Expected unique bestval: " + str(directory))
    checkpoint(files[0])
    return files[0]


def valid_completion(row, stage, record):
    require(record.get("exit_code") == 0, "Completed stage lacks zero exit code")
    directory = OUTPUT / row / ("final_infer" if stage == "merge" else stage)
    require(record["config_sha256"] == digest(directory / "config.yaml"), "Completed config drift")
    for item in record["artifacts"]:
        path = Path(item["path"])
        require(path.is_file() and digest(path) == item["sha256"], "Artifact missing/changed: " + str(path))
    if stage == "final_infer":
        require(tuple(record["ap"]) == parse_ap(Path(record["log"]).read_text(errors="replace")),
                "Inference AP no longer matches its log")
    else:
        checkpoint(record["checkpoint"])
    return True


def execute(command, directory, manifest, stage):
    directory.mkdir(parents=True, exist_ok=True)
    attempts = manifest["stages"].get(stage, {}).get("attempts", [])
    log = directory / ("%s.attempt%d.log" % (stage.replace("/", "_"), len(attempts) + 1))
    attempt = dict(command=[str(x) for x in command], log=str(log), started=time.time())
    attempts.append(attempt)
    manifest["status"] = "RUNNING"
    manifest["current_stage"] = stage
    manifest["stages"][stage] = dict(status="RUNNING", attempts=attempts)
    write_json(OUTPUT / manifest["experiment"] / "run_manifest.json", manifest)
    print("START", manifest["experiment"], stage, "log=", log, flush=True)
    with log.open("x", encoding="utf8") as stream:
        process = subprocess.Popen([str(x) for x in command], cwd=str(ROOT), stdout=stream,
                                   stderr=subprocess.STDOUT)
        attempt["pid"] = process.pid
        write_json(OUTPUT / manifest["experiment"] / "run_manifest.json", manifest)
        code = process.wait()
    attempt.update(exit_code=code, finished=time.time())
    write_json(OUTPUT / manifest["experiment"] / "run_manifest.json", manifest)
    require(code == 0, "Stage %s exit=%s; log=%s" % (stage, code, log))
    return log


def complete_record(manifest, stage, directory, log, ckpt=None, ap=None):
    artifacts = [log]
    if ckpt:
        artifacts.append(ckpt)
    record = manifest["stages"][stage]
    record.update(status="COMPLETE", exit_code=0, log=str(log), finished=time.time(),
                  config_sha256=digest(directory / "config.yaml"),
                  artifacts=[dict(path=str(x), sha256=digest(x)) for x in artifacts])
    if ckpt:
        record["checkpoint"] = str(ckpt)
    if ap:
        record["ap"] = ap
    write_json(OUTPUT / manifest["experiment"] / "run_manifest.json", manifest)


def train_stage(row, stage, manifest):
    manifest["current_stage"] = stage
    directory = OUTPUT / row / stage
    existing = manifest["stages"].get(stage, {})
    if existing.get("status") == "COMPLETE":
        valid_completion(row, stage, existing)
        print("SKIP verified", row, stage, flush=True)
        return
    resume = directory / "resume_state.pth"
    if not resume.exists():
        # Legacy weights alone do not restore Adam/scheduler/RNG. Archive failed
        # attempt files and replay this stage from its deterministic initialization.
        leftovers = list(directory.glob("net_epoch*.pth"))
        if leftovers:
            archive = directory / ("interrupted_%d" % time.time_ns())
            archive.mkdir()
            for path in leftovers:
                path.rename(archive / path.name)
        if stage.startswith("stage2/"):
            anchor = best(OUTPUT / row / "stage1")
            # Historical Stage2 starts at init_epoch=1. Copy, never symlink: a
            # numbered checkpoint writer must not overwrite the Stage1 anchor.
            shutil.copy2(anchor, directory / "net_epoch1.pth")
            manifest["stages"].setdefault(stage, {})["anchor"] = dict(path=str(anchor), sha256=digest(anchor))
    gpu_idle()
    log = execute([sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
                   "--nproc_per_node=2", "--module", "opencood.tools.train_table4_ext",
                   "--hypes_yaml", directory / "config.yaml", "--model_dir", directory],
                  directory, manifest, stage)
    from opencood.tools.table4_ext_state import load
    saved = load(resume)
    cfg = yaml.safe_load((directory / "config.yaml").read_text())
    require(saved["next_epoch"] == cfg["train_params"]["epoches"], "Incomplete epoch budget")
    require("Training Finished" in log.read_text(errors="replace"), "Missing training completion line")
    selected = best(directory)
    require(selected.name == "net_epoch_bestval_at%d.pth" % saved["best_epoch"], "Bestval state mismatch")
    complete_record(manifest, stage, directory, log, ckpt=selected)
    record = manifest["stages"][stage]
    record.update(bestval_epoch=saved["best_epoch"], bestval_loss=saved["best_loss"],
                  next_epoch=saved["next_epoch"], fully_trained=True,
                  resume_state_sha256=digest(resume))
    if stage.startswith("stage2/"):
        anchor = best(OUTPUT / row / "stage1")
        record["anchor"] = dict(path=str(anchor), sha256=digest(anchor))
    write_json(OUTPUT / row / "run_manifest.json", manifest)


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def worker():
    import fcntl
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another worker holds this experiment lock")
        gpu_idle()
        prepare()
        write_json(OUTPUT / "worker.json", dict(pid=os.getpid(), started=time.time(),
                   boot_id=boot_id()))
        for row in ROWS:
            manifest = read_json(OUTPUT / row / "run_manifest.json")
            try:
                for stage in STAGES[:-1]:
                    train_stage(row, stage, manifest)
                directory = OUTPUT / row / "final_infer"
                merged = manifest["stages"].get("merge", {})
                if merged.get("status") == "COMPLETE":
                    valid_completion(row, "merge", merged)
                else:
                    manifest["current_stage"] = "merge"
                    log = execute([sys.executable, "-B", "-m", "opencood.tools.doma_tools", "merge_final",
                                   OUTPUT / row / "stage2/m2", OUTPUT / row / "stage2/m3",
                                   OUTPUT / row / "stage2/m4", OUTPUT / row / "stage1", directory],
                                  directory, manifest, "merge")
                    checkpoint(directory / "net_epoch1.pth")
                    run([sys.executable, "-B", "-m", "opencood.tools.check_table4_ext",
                         "--merged", directory], timeout=300)
                    complete_record(manifest, "merge", directory, log, ckpt=directory / "net_epoch1.pth")
                inferred = manifest["stages"].get("final_infer", {})
                if inferred.get("status") == "COMPLETE":
                    valid_completion(row, "final_infer", inferred)
                else:
                    manifest["current_stage"] = "final_infer"
                    gpu_idle()
                    log = execute([sys.executable, "-B", "-m", "opencood.tools.inference_doma_heter_in_order",
                                   "--model_dir", directory, "--fusion_method", "intermediate",
                                   "--use_cav", "[1,2,3,4]", "--range", "204.8,102.4"],
                                  directory, manifest, "final_infer")
                    ap = parse_ap(log.read_text(errors="replace"))
                    shutil.copy2(log, directory / "infer.log")
                    complete_record(manifest, "final_infer", directory, directory / "infer.log", ap=ap)
                manifest.update(status="COMPLETE", finished=time.time())
                manifest.pop("error", None)
                write_json(OUTPUT / row / "run_manifest.json", manifest)
                summarize()
            except BaseException as error:
                manifest.update(status="FAILED", error=str(error), failed=time.time())
                stage = manifest.get("current_stage")
                if stage in manifest["stages"]:
                    manifest["stages"][stage]["status"] = "FAILED"
                write_json(OUTPUT / row / "run_manifest.json", manifest)
                summarize()
                raise


def summarize():
    rows = []
    for row, flags in ROWS.items():
        path = OUTPUT / row / "run_manifest.json"
        m = read_json(path) if path.exists() else {"status": "PENDING", "stages": {}}
        record = m["stages"].get("final_infer", {})
        ap = ["", "", ""]
        if m["status"] == "COMPLETE":
            valid_completion(row, "final_infer", record)
            ap = list(record["ap"])
        rows.append(dict(experiment=row, M=flags[0], G=flags[1], C=flags[2],
                         AP30=ap[0], AP50=ap[1], AP70=ap[2], status=m["status"],
                         source=record.get("log", ""), delta50_vs_full="", delta70_vs_full=""))
    # Historical values are explicitly snapshot evidence; not newly measured AP.
    snapshot = read_json(CONFIGS / "history_snapshot.json")
    sources = [("Base_object_branch", (0, 0, 0), HISTORY + "Row1_Base/final_infer/infer.log"),
               ("M_only_historical", (1, 0, 0), HISTORY + "Row2_DualLayer/final_infer/infer.log"),
               ("M_G_historical", (1, 1, 0), HISTORY + "Row3_Geometry/final_infer/infer.log"),
               ("Full_historical", (1, 1, 1), FULL + "m2lr0p0012_m3lr0p0030_m4lr0p0017--final-result/infer.log")]
    full_ap = parse_ap(snapshot[sources[-1][2]])
    for row in rows:
        if row["status"] == "COMPLETE":
            row["delta50_vs_full"] = round(row["AP50"] - full_ap[1], 4)
            row["delta70_vs_full"] = round(row["AP70"] - full_ap[2], 4)
    for name, flags, source in sources:
        ap = parse_ap(snapshot[source])
        rows.append(dict(experiment=name, M=flags[0], G=flags[1], C=flags[2], AP30=ap[0],
                         AP50=ap[1], AP70=ap[2], status="HISTORICAL_LOG_ONLY_PROTOCOL_DIFF",
                         source=source, delta50_vs_full="", delta70_vs_full=""))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(OUTPUT / "ablation_results.csv", stream.getvalue())
    keys = list(rows[0])
    text = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    text += ["| " + " | ".join(str(row[k]) for k in keys) + " |" for row in rows]
    text += ["", "Historical Full uses a different Stage2 budget/checkpoint selection history.",
             "Deltas are descriptive, not a controlled causal comparison.",
             "HEAL 0.8922/0.8103 was supplied by the user; source unverified and excluded."]
    atomic_text(OUTPUT / "ablation_results.md", "\n".join(text) + "\n")
    report = (ROOT / "docs/DOMA_TABLE4_EXT.md").read_text(encoding="utf8")
    atomic_text(OUTPUT / "execution_report.md", report + "\n\nCurrent results:\n\n" + "\n".join(text))


def status():
    active = False
    if sys.platform == "linux" and (OUTPUT / ".worker.lock").exists():
        import fcntl
        with (OUTPUT / ".worker.lock").open("r") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                active = True
    print("Worker lock active:", active)
    controllers = sorted((OUTPUT / "logs").glob("controller-*.log"))
    if controllers:
        print("Controller log:", controllers[-1])
        with controllers[-1].open("rb") as stream:
            stream.seek(max(0, controllers[-1].stat().st_size - 2048))
            lines = stream.read().decode("utf8", errors="replace").splitlines()
        print("Controller last:", lines[-1] if lines else "(empty)")
    for row in ROWS:
        path = OUTPUT / row / "run_manifest.json"
        if not path.exists():
            print(row, "PENDING")
            continue
        m = read_json(path)
        state = m["status"]
        if state == "RUNNING" and not active:
            state = "FAILED (interrupted/stale RUNNING; resume required)"
        print(row, state, m.get("current_stage", ""), m.get("error", ""))
        rec = m.get("stages", {}).get(m.get("current_stage"), {})
        attempts = rec.get("attempts", [])
        if attempts:
            print("  process:", attempts[-1].get("pid"), "exit:", attempts[-1].get("exit_code"))
            log = Path(attempts[-1]["log"])
            if log.exists():
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 4096))
                    lines = stream.read().decode("utf8", errors="replace").splitlines()
                print("  last log:", lines[-1] if lines else "(empty)")
    if shutil.which("nvidia-smi"):
        run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "status", "prepare", "worker"))
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.mode == "check":
        check(args.local)
    elif args.mode == "status":
        status()
    elif args.mode == "prepare":
        prepare()
    else:
        worker()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[TABLE4 EXT FAILED]", exc, file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
