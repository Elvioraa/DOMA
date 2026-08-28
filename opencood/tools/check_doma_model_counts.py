"""Instantiate the m1 Stage1 models and verify full parameter/state counts.

This is an import-only CPU diagnostic for environments without optional CUDA,
spconv, camera, or geometry packages.  Stubs are installed only for missing
packages and are never used by a forward pass.  Do not use this script for m3
or any model whose construction genuinely depends on those implementations.
"""

import json
import sys
import types

import torch.nn as nn


EXPECTED = {
    "Official HEAL": (5464791, 5464791, 382, 0),
    "DOMA V1": (5603039, 5603039, 406, 138248),
    "DOMA V2": (5780320, 5780320, 423, 315529),
    "DOMA V3": (5790817, 5790817, 427, 326026),
}


class _DummySparse(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, value):
        return value


class _DropPath(nn.Identity):
    def __init__(self, drop_prob=0.0):
        super().__init__()


def _install_optional_import_stubs():
    """Provide inert construction-time symbols only when a package is absent."""
    try:
        import icecream  # noqa: F401
    except ImportError:
        module = types.ModuleType("icecream")
        module.ic = lambda *args, **kwargs: args[0] if len(args) == 1 else args
        sys.modules["icecream"] = module

    try:
        import spconv  # noqa: F401
    except ImportError:
        module = types.ModuleType("spconv")
        for name in (
            "SparseSequential",
            "SubMConv3d",
            "SparseConv3d",
            "SparseInverseConv3d",
            "SparseConvTensor",
        ):
            setattr(module, name, _DummySparse)
        sys.modules["spconv"] = module

    try:
        from timm.models.layers import DropPath  # noqa: F401
    except ImportError:
        timm = types.ModuleType("timm")
        models = types.ModuleType("timm.models")
        layers = types.ModuleType("timm.models.layers")
        layers.DropPath = _DropPath
        sys.modules.update(
            {"timm": timm, "timm.models": models, "timm.models.layers": layers}
        )

    try:
        import einops  # noqa: F401
    except ImportError:
        module = types.ModuleType("einops")
        module.rearrange = lambda value, *args, **kwargs: value
        module.repeat = lambda value, *args, **kwargs: value
        sys.modules["einops"] = module

    try:
        import efficientnet_pytorch  # noqa: F401
    except ImportError:
        module = types.ModuleType("efficientnet_pytorch")

        class EfficientNet(nn.Module):
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        module.EfficientNet = EfficientNet
        sys.modules["efficientnet_pytorch"] = module

    try:
        import shapely.geometry  # noqa: F401
    except ImportError:
        shapely = types.ModuleType("shapely")
        geometry = types.ModuleType("shapely.geometry")
        for name in ("Polygon", "Point", "MultiPoint"):
            setattr(geometry, name, type(name, (), {}))
        sys.modules.update({"shapely": shapely, "shapely.geometry": geometry})

    try:
        import pyquaternion  # noqa: F401
    except ImportError:
        module = types.ModuleType("pyquaternion")
        module.Quaternion = type("Quaternion", (), {})
        sys.modules["pyquaternion"] = module


def _counts(model):
    parameters = tuple(model.named_parameters())
    return {
        "parameters": sum(parameter.numel() for _, parameter in parameters),
        "trainable_parameters": sum(
            parameter.numel()
            for _, parameter in parameters
            if parameter.requires_grad
        ),
        "state_tensors": len(model.state_dict()),
        "doma_parameters": sum(
            parameter.numel()
            for name, parameter in parameters
            if name.startswith("doma_")
        ),
    }


def main():
    _install_optional_import_stubs()
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.models.doma_heter_pyramid_collab import DOMAHeterPyramidCollab
    from opencood.models.heter_pyramid_collab import HeterPyramidCollab

    official_hypes = load_yaml(
        "opencood/hypes_yaml/opv2v/MoreModality/HEAL/stage1/m1_pyramid.yaml"
    )
    models = {
        "Official HEAL": HeterPyramidCollab(official_hypes["model"]["args"]),
    }
    for version in ("V1", "V2", "V3"):
        hypes = load_yaml(
            "opencood/hypes_yaml/opv2v/MoreModality/DOMA/%s/stage1/m1.yaml"
            % version
        )
        models["DOMA %s" % version] = DOMAHeterPyramidCollab(
            hypes["model"]["args"]
        )

    report = {name: _counts(model) for name, model in models.items()}
    for name, expected in EXPECTED.items():
        actual = report[name]
        observed = (
            actual["parameters"],
            actual["trainable_parameters"],
            actual["state_tensors"],
            actual["doma_parameters"],
        )
        if observed != expected:
            raise AssertionError("%s count mismatch: %r != %r" % (name, observed, expected))
    print(json.dumps(report, indent=2, sort_keys=True))
    print("DOMA full-model count acceptance: PASS")


if __name__ == "__main__":
    main()
