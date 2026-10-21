"""Read-only acceptance checks for the V2XSet DOMA V2 configuration family."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DOMA_ROOT = (
    REPO_ROOT
    / "opencood"
    / "hypes_yaml"
    / "v2xset"
    / "MoreModality"
    / "DOMA"
    / "V2"
)
HEAL_ROOT = (
    REPO_ROOT
    / "opencood"
    / "hypes_yaml"
    / "v2xset"
    / "MoreModality"
    / "HEAL"
)
OPV2V_DOMA_ROOT = (
    REPO_ROOT
    / "opencood"
    / "hypes_yaml"
    / "opv2v"
    / "MoreModality"
    / "DOMA"
    / "V2"
)

CONFIG_SPECS = {
    "stage1/m1.yaml": {
        "reference": "stage1/m1_pyramid.yaml",
        "model": "doma_heter_pyramid_collab",
        "loss": "doma_point_pillar_pyramid_loss",
        "mode": "stage1_anchor",
        "active_modality": None,
    },
    "stage2/m2.yaml": {
        "reference": "stage2/m2_single_pyramid.yaml",
        "model": "doma_heter_pyramid_single",
        "loss": "doma_point_pillar_pyramid_loss",
        "mode": "stage2_adapt",
        "active_modality": "m2",
    },
    "stage2/m3.yaml": {
        "reference": "stage2/m3_single_pyramid.yaml",
        "model": "doma_heter_pyramid_single",
        "loss": "doma_point_pillar_pyramid_loss",
        "mode": "stage2_adapt",
        "active_modality": "m3",
    },
    "stage2/m4.yaml": {
        "reference": "stage2/m4_single_pyramid.yaml",
        "model": "doma_heter_pyramid_single",
        "loss": "doma_point_pillar_pyramid_loss",
        "mode": "stage2_adapt",
        "active_modality": "m4",
    },
    "final_infer/m1m2m3m4.yaml": {
        "reference": "final_infer/m1m2m3m4.yaml",
        "model": "doma_heter_pyramid_collab",
        "loss": "point_pillar_pyramid_loss",
        "mode": "inference",
        "active_modality": None,
    },
}

EXPECTED_DATA_PATHS = {
    "root_dir": "dataset/V2XSET/train",
    "validate_dir": "dataset/V2XSET/validate",
    "test_dir": "dataset/V2XSET/test",
}
EXPECTED_ASSIGNMENT = (
    "opencood/logs/heter_modality_assign/v2xset_4modality.json"
)
VALID_MODALITIES = {"m1", "m2", "m3", "m4"}
CAMERA_FILTERED_SCENARIO = "2021_09_09_13_20_58"


class Audit:
    def __init__(self):
        self.failures = []
        self.notes = []

    def require(self, condition, message):
        if not condition:
            self.failures.append(message)

    def note(self, message):
        self.notes.append(message)

    def finish(self):
        for message in self.notes:
            print("INFO:", message)
        if self.failures:
            for message in self.failures:
                print("FAIL:", message)
            raise SystemExit(
                "V2XSet DOMA adaptation check failed with %d error(s)"
                % len(self.failures)
            )
        print("V2XSet DOMA adaptation check: PASS")


def load_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise TypeError("%s must contain a YAML mapping" % path)
    return data


def normalized_reference_candidate(candidate, reference):
    """Remove the only intentional DOMA-vs-HEAL differences."""
    normalized = copy.deepcopy(candidate)
    normalized["name"] = reference["name"]
    normalized["model"]["core_method"] = reference["model"]["core_method"]
    normalized["model"]["args"].pop("doma", None)
    normalized["loss"]["core_method"] = reference["loss"]["core_method"]
    return normalized


def normalized_doma_method(doma):
    normalized = copy.deepcopy(doma)
    normalized.pop("mode", None)
    normalized.pop("active_modality", None)
    return normalized


def check_configs(audit):
    loaded = {}
    method_contract = None
    for relative, spec in CONFIG_SPECS.items():
        path = DOMA_ROOT / relative
        reference_path = HEAL_ROOT / spec["reference"]
        opv2v_doma_path = OPV2V_DOMA_ROOT / relative
        audit.require(path.is_file(), "missing config: %s" % path)
        audit.require(
            reference_path.is_file(), "missing HEAL reference: %s" % reference_path
        )
        audit.require(
            opv2v_doma_path.is_file(),
            "missing OPV2V DOMA V2 method reference: %s" % opv2v_doma_path,
        )
        if (
            not path.is_file()
            or not reference_path.is_file()
            or not opv2v_doma_path.is_file()
        ):
            continue

        raw = path.read_text(encoding="utf-8")
        audit.require(
            "OPV2V" not in raw and "opv2v_4modality" not in raw,
            "%s contains an OPV2V path or assignment reference" % relative,
        )
        config = load_yaml(path)
        reference = load_yaml(reference_path)
        opv2v_doma = load_yaml(opv2v_doma_path)
        loaded[relative] = config

        for key, expected in EXPECTED_DATA_PATHS.items():
            audit.require(
                config.get(key) == expected,
                "%s has unexpected %s=%r" % (relative, key, config.get(key)),
            )
        audit.require(
            config.get("fusion", {}).get("dataset") == "v2xset",
            "%s must use fusion.dataset=v2xset" % relative,
        )
        audit.require(
            config.get("heter", {}).get("assignment_path") == EXPECTED_ASSIGNMENT,
            "%s has the wrong modality assignment path" % relative,
        )
        audit.require(
            config.get("model", {}).get("core_method") == spec["model"],
            "%s has the wrong DOMA model entry" % relative,
        )
        audit.require(
            config.get("loss", {}).get("core_method") == spec["loss"],
            "%s has the wrong loss entry" % relative,
        )

        doma = config.get("model", {}).get("args", {}).get("doma")
        audit.require(isinstance(doma, dict), "%s has no DOMA method block" % relative)
        if isinstance(doma, dict):
            audit.require(
                doma.get("enabled") is True and doma.get("version") == "v2",
                "%s must enable canonical DOMA V2" % relative,
            )
            audit.require(
                doma.get("mode") == spec["mode"],
                "%s has the wrong DOMA mode" % relative,
            )
            audit.require(
                doma.get("active_modality") == spec["active_modality"],
                "%s has the wrong active modality" % relative,
            )
            audit.require(
                doma.get("refiner", {}).get("yaw_mode") == "sin_cos_centered",
                "%s must use the canonical V2 centered-yaw contract" % relative,
            )
            current_contract = normalized_doma_method(doma)
            opv2v_contract = normalized_doma_method(
                opv2v_doma["model"]["args"]["doma"]
            )
            audit.require(
                current_contract == opv2v_contract,
                "%s changes the canonical OPV2V DOMA V2 method contract"
                % relative,
            )
            if method_contract is None:
                method_contract = current_contract
            else:
                audit.require(
                    current_contract == method_contract,
                    "%s differs from the shared DOMA merge fingerprint" % relative,
                )

        audit.require(
            normalized_reference_candidate(config, reference) == reference,
            "%s changes V2XSet HEAL data/sensor/training/evaluation settings"
            % relative,
        )

    m3 = loaded.get("stage2/m3.yaml", {})
    final = loaded.get("final_infer/m1m2m3m4.yaml", {})
    audit.require(
        "lidar_channels_dict" not in m3.get("heter", {}),
        "V2XSet m3 must not request OPV2V_Hetero _32.pcd files",
    )
    audit.require(
        "lidar_channels_dict" not in final.get("heter", {}),
        "V2XSet final inference must not request OPV2V_Hetero _32.pcd files",
    )
    audit.require(
        final.get("input_source") == ["lidar", "camera"],
        "V2XSet final inference must preserve the no-depth input protocol",
    )
    for modality in ("m2", "m4"):
        encoder = final.get("model", {}).get("args", {}).get(modality, {}).get(
            "encoder_args", {}
        )
        audit.require(
            encoder.get("depth_supervision") is False,
            "V2XSet final %s must keep depth supervision disabled" % modality,
        )


def check_source_contracts(audit):
    dataset_registry = (
        REPO_ROOT / "opencood" / "data_utils" / "datasets" / "__init__.py"
    ).read_text(encoding="utf-8")
    v2xset_dataset = (
        REPO_ROOT
        / "opencood"
        / "data_utils"
        / "datasets"
        / "basedataset"
        / "v2xset_basedataset.py"
    ).read_text(encoding="utf-8")
    heter_utils = (
        REPO_ROOT / "opencood" / "utils" / "heter_utils.py"
    ).read_text(encoding="utf-8")
    official_infer = (
        REPO_ROOT / "opencood" / "tools" / "inference_heter_in_order.py"
    ).read_text(encoding="utf-8")
    doma_infer = (
        REPO_ROOT / "opencood" / "tools" / "inference_doma_heter_in_order.py"
    ).read_text(encoding="utf-8")
    doma_tools = (
        REPO_ROOT / "opencood" / "tools" / "doma_tools.py"
    ).read_text(encoding="utf-8")

    audit.require(
        "V2XSETBaseDataset" in dataset_registry
        and "'v2xset'" in dataset_registry,
        "V2XSet is not registered in build_dataset",
    )
    audit.require(
        "class V2XSETBaseDataset(OPV2VBaseDataset)" in v2xset_dataset,
        "V2XSETBaseDataset no longer reuses the OPV2V loader",
    )
    audit.require(
        CAMERA_FILTERED_SCENARIO in v2xset_dataset,
        "the known incomplete-camera V2XSet scenario filter is missing",
    )
    audit.require(
        "def assign_modality_4(" in heter_utils
        and "def assign_modality_4_in_order(" in heter_utils,
        "four-modality assignment generators are missing",
    )
    audit.require(
        "if cav_list[0] == '-1':" in heter_utils,
        "in-order assignment no longer moves infrastructure behind vehicles",
    )
    for token in (
        "hypes['validate_dir'] = hypes['test_dir']",
        "hypes['comm_range'] = 180",
        'replace(".json", "_in_order.json")',
        "eval_utils.eval_final_results",
    ):
        audit.require(
            token in official_infer and token in doma_infer,
            "HEAL/DOMA ordered inference protocol differs at: %s" % token,
        )
    audit.require(
        "inference_intermediate_fusion_doma" in doma_infer
        and "refine_doma_detections" in doma_infer,
        "DOMA ordered inference entry is incomplete",
    )
    audit.require(
        "merge_final" in doma_tools
        and '("m2", "m3", "m4")' in doma_tools
        and "apply_doma_merge_ownership" in doma_tools,
        "DOMA four-checkpoint final merge support is missing",
    )


def scenario_cavs(split_path):
    return {
        scenario.name: sorted(
            child.name for child in scenario.iterdir() if child.is_dir()
        )
        for scenario in sorted(split_path.iterdir())
        if scenario.is_dir()
    }


def dataset_sample_count(split_path, camera_compatible):
    total = 0
    for scenario in sorted(split_path.iterdir()):
        if not scenario.is_dir():
            continue
        if camera_compatible and scenario.name == CAMERA_FILTERED_SCENARIO:
            continue
        cavs = sorted(child for child in scenario.iterdir() if child.is_dir())
        if not cavs:
            continue
        if int(cavs[0].name) < 0:
            cavs = cavs[1:] + cavs[:1]
        timestamps = [
            path
            for path in cavs[0].glob("*.yaml")
            if "additional" not in path.name
            and not (
                "2021_08_20_21_10_24" in str(path)
                and path.stem == "000265"
            )
        ]
        total += len(timestamps)
    return total


def check_sample_files(audit, scenario, cav):
    yaml_files = sorted(
        path
        for path in cav.glob("*.yaml")
        if "additional" not in path.name
    )
    audit.require(bool(yaml_files), "no timestamp YAML files in %s" % cav)
    if not yaml_files:
        return
    metadata_path = yaml_files[0]
    timestamp = metadata_path.stem
    metadata = load_yaml(metadata_path)
    audit.require(
        isinstance(metadata.get("vehicles"), dict),
        "%s has no vehicle label mapping" % metadata_path,
    )
    audit.require(
        "lidar_pose" in metadata or "lidar_pose_clean" in metadata,
        "%s has no LiDAR pose" % metadata_path,
    )
    audit.require(
        (cav / (timestamp + ".pcd")).is_file(),
        "%s is missing its LiDAR point cloud" % metadata_path,
    )
    for camera_id in range(4):
        image = cav / ("%s_camera%d.png" % (timestamp, camera_id))
        calibration = metadata.get("camera%d" % camera_id)
        audit.require(image.is_file(), "missing camera image: %s" % image)
        audit.require(
            isinstance(calibration, dict)
            and "cords" in calibration
            and "intrinsic" in calibration,
            "%s lacks camera%d calibration" % (metadata_path, camera_id),
        )
    audit.note(
        "sample %s/%s/%s: timestamp=%s labels=%d"
        % (
            scenario.parent.name,
            scenario.name,
            cav.name,
            timestamp,
            len(metadata.get("vehicles", {})),
        )
    )


def load_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def check_assignments(audit, data_root, assignment_dir, split_cavs):
    random_path = assignment_dir / "v2xset_4modality.json"
    ordered_path = assignment_dir / "v2xset_4modality_in_order.json"
    audit.require(random_path.is_file(), "missing assignment: %s" % random_path)
    audit.require(ordered_path.is_file(), "missing assignment: %s" % ordered_path)
    if random_path.is_file():
        assignment = load_json(random_path)
        for split in ("train", "validate", "test"):
            for scenario, cavs in split_cavs[split].items():
                actual = assignment.get(scenario)
                audit.require(
                    isinstance(actual, dict) and sorted(actual) == cavs,
                    "%s does not cover every %s CAV" % (random_path, scenario),
                )
                if isinstance(actual, dict):
                    audit.require(
                        set(actual.values()) <= VALID_MODALITIES,
                        "%s has an illegal modality in %s" % (random_path, scenario),
                    )
    if ordered_path.is_file():
        assignment = load_json(ordered_path)
        test_scenarios = split_cavs["test"]
        audit.require(
            sorted(assignment) == sorted(test_scenarios),
            "%s must contain exactly the test scenarios" % ordered_path,
        )
        for scenario, cavs in test_scenarios.items():
            ordered_cavs = list(cavs)
            if ordered_cavs and ordered_cavs[0] == "-1":
                ordered_cavs = ordered_cavs[1:] + ordered_cavs[:1]
            expected = {
                cav_id: "m%d" % (index % 4 + 1)
                for index, cav_id in enumerate(ordered_cavs)
            }
            actual = assignment.get(scenario)
            audit.require(
                actual == expected,
                "%s has the wrong in-order mapping for %s"
                % (ordered_path, scenario),
            )
            if ordered_cavs:
                audit.require(
                    actual.get(ordered_cavs[0]) == "m1",
                    "%s does not assign the ego vehicle to m1" % scenario,
                )


def check_data(audit, data_root, assignment_dir, sample_scenarios):
    split_cavs = {}
    for split in ("train", "validate", "test"):
        split_path = data_root / split
        audit.require(split_path.is_dir(), "missing data split: %s" % split_path)
        if not split_path.is_dir():
            continue
        split_cavs[split] = scenario_cavs(split_path)
        audit.require(
            bool(split_cavs[split]), "no scenarios found in %s" % split_path
        )
        lidar_count = dataset_sample_count(split_path, camera_compatible=False)
        camera_count = dataset_sample_count(split_path, camera_compatible=True)
        audit.note(
            "%s: scenarios=%d lidar_samples=%d camera_compatible_samples=%d"
            % (split, len(split_cavs[split]), lidar_count, camera_count)
        )
        if CAMERA_FILTERED_SCENARIO in split_cavs[split]:
            audit.note(
                "%s/%s is intentionally excluded by V2XSETBaseDataset "
                "when camera input is enabled"
                % (split, CAMERA_FILTERED_SCENARIO)
            )
        sample_names = [
            scenario_name
            for scenario_name in sorted(split_cavs[split])
            if scenario_name != CAMERA_FILTERED_SCENARIO
        ][:sample_scenarios]
        for scenario_name in sample_names:
            scenario = split_path / scenario_name
            cavs = [scenario / cav_id for cav_id in split_cavs[split][scenario_name]]
            audit.require(bool(cavs), "no CAV directories in %s" % scenario)
            for cav in cavs:
                check_sample_files(audit, scenario, cav)
    if set(split_cavs) == {"train", "validate", "test"}:
        check_assignments(audit, data_root, assignment_dir, split_cavs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read-only checks for the V2XSet DOMA V2 adaptation"
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="also require and sample the real V2XSet data and assignment JSON files",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "V2XSET",
    )
    parser.add_argument(
        "--assignment-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "opencood"
            / "logs"
            / "heter_modality_assign"
        ),
    )
    parser.add_argument("--sample-scenarios", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sample_scenarios < 1:
        raise SystemExit("--sample-scenarios must be positive")
    audit = Audit()
    check_configs(audit)
    check_source_contracts(audit)
    if args.check_data:
        check_data(
            audit,
            args.data_root.resolve(),
            args.assignment_dir.resolve(),
            args.sample_scenarios,
        )
    elif not args.data_root.is_dir():
        audit.note(
            "real V2XSet data not present locally; data and assignment checks are UNVERIFIED"
        )
    else:
        audit.note("real data exists but was not scanned; rerun with --check-data")
    audit.finish()


if __name__ == "__main__":
    main()
