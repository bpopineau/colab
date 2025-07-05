import sys
import types
import runpy

import pytest


def dummy_module(**attrs):
    mod = types.ModuleType("dummy")
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def test_help(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        dummy_module(
            cuda=dummy_module(is_available=lambda: False),
            device=lambda *a, **k: None,
            load=lambda *a, **k: {},
        ),
    )
    monkeypatch.setitem(sys.modules, "numpy", dummy_module())
    monkeypatch.setitem(sys.modules, "PIL", dummy_module(Image=object))
    monkeypatch.setitem(
        sys.modules,
        "torchvision.transforms",
        dummy_module(ToTensor=lambda: None, ToPILImage=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules, "basicsr.archs.rrdbnet_arch", dummy_module(RRDBNet=object)
    )
    monkeypatch.setitem(sys.modules, "realesrgan", dummy_module(RealESRGANer=object))
    monkeypatch.setitem(sys.modules, "mirnet.model", dummy_module(MIRNet=object))

    monkeypatch.setattr(sys, "argv", ["inference.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path("photo-restoration-pipeline/inference.py", run_name="__main__")
    assert exc.value.code == 0
