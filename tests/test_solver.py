"""Solver through real ONNX Runtime, with tiny hand-built models for both decoders."""

from __future__ import annotations

import json

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from captcha2text.model import Solver  # noqa: E402


def _save(folder, graph, spec):
    folder.mkdir(parents=True, exist_ok=True)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(folder / "model.onnx"))
    (folder / "model.json").write_text(json.dumps(spec, ensure_ascii=False))
    return Solver(folder)


def _constant_graph(inp, outputs: dict[str, np.ndarray]):
    """Outputs are constants plus 0 * sum(image), so the input is still used."""
    nodes, outs = [], []
    nodes.append(helper.make_node("ReduceSum", ["image"], ["s"], keepdims=0))
    nodes.append(helper.make_node("Mul", ["s", "zero"], ["z"]))
    inits = [numpy_helper.from_array(np.array(0, dtype=np.float32), "zero")]
    for name, value in outputs.items():
        inits.append(numpy_helper.from_array(value, f"{name}_c"))
        if value.dtype == np.float32:
            nodes.append(helper.make_node("Add", [f"{name}_c", "z"], [name]))
        else:
            nodes.append(helper.make_node("Identity", [f"{name}_c"], [name]))
        dtype = TensorProto.FLOAT if value.dtype == np.float32 else TensorProto.INT64
        outs.append(helper.make_tensor_value_info(name, dtype, list(value.shape)))
    return helper.make_graph(nodes, "t", [inp], outs, initializer=inits)


def test_ctc_model(tmp_path):
    charset = "ab"
    path = [1, 1, 0, 2, 0, 2]  # a a _ b _ b -> "abb"
    logits = np.zeros((1, len(path), 3), dtype=np.float32)
    for t, k in enumerate(path):
        logits[0, t, k] = 12.0
    inp = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 48, 128])
    solver = _save(
        tmp_path,
        _constant_graph(inp, {"logits": logits}),
        {"name": "t-ctc", "input": {"width": 128, "height": 48, "channels": 3, "layout": "NCHW"},
         "decoder": "ctc", "charset": charset},
    )
    result = solver.solve(np.full((60, 160, 3), 200, dtype=np.uint8))
    assert solver.name == "t-ctc"
    assert result.text == "abb"
    assert 0.99 < result.confidence <= 1.0


def test_ids_model(tmp_path):
    tokens = ["", "<start>", "<end>", "k", "e", "r", "y"]
    ids = np.array([3, 4, 5, 5, 6, 2, 2], dtype=np.int64)
    probs = np.array([0.9, 1, 1, 1, 1, 0.9, 1], dtype=np.float32)
    inp = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 50, 100, 3])
    solver = _save(
        tmp_path,
        _constant_graph(inp, {"ids": ids, "probs": probs}),
        {"name": "t-ids", "input": {"width": 100, "height": 50, "channels": 3, "layout": "NHWC"},
         "decoder": "ids", "tokens": tokens},
    )
    result = solver.solve(np.zeros((50, 100, 3), dtype=np.uint8))
    assert result.text == "kerry"
    assert result.confidence == pytest.approx(0.81, abs=1e-4)


def _path_logits(path_classes, n_classes):
    logits = np.zeros((len(path_classes), n_classes), dtype=np.float32)
    for t, k in enumerate(path_classes):
        logits[t, k] = 12.0
    return logits


def _ddddocr_like(folder, logits_tc, charset, **spec_extra):
    """Time-major (T, 1, C) output, 1-channel input of any width, like ddddocr's models."""
    logits = logits_tc[:, None, :].astype(np.float32)
    inp = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 1, 64, "w"])
    nodes = [
        helper.make_node("ReduceSum", ["input1"], ["s"], keepdims=0),
        helper.make_node("Mul", ["s", "zero"], ["z"]),
        helper.make_node("Add", ["c", "z"], ["387"]),
    ]
    inits = [numpy_helper.from_array(np.array(0, np.float32), "zero"), numpy_helper.from_array(logits, "c")]
    out = helper.make_tensor_value_info("387", TensorProto.FLOAT, list(logits.shape))
    graph = helper.make_graph(nodes, "d", [inp], [out], initializer=inits)
    spec = {
        "name": "dd-test",
        "input": {"width": None, "height": 64, "channels": 1, "layout": "NCHW", "resample": "lanczos",
                  "mean": 0.5, "std": 0.5},
        "decoder": "ctc", "output_layout": "TNC", "charset": charset, **spec_extra,
    }
    return _save(folder, graph, spec)


def test_ddddocr_style_model(tmp_path):
    charset = ["''", "X", "z", "V", "中"]  # a multi-character entry, like ddddocr's charset
    solver = _ddddocr_like(tmp_path, _path_logits([2, 0, 3, 3, 4, 0, 1], 6), charset)
    result = solver.solve(np.full((60, 160, 3), 230, dtype=np.uint8))
    assert result.text == "XzV''"


def test_allowed_picks_best_allowed_class_then_lowercases(tmp_path):
    charset = ["a", "B", "中"]  # classes: 0 blank, 1 a, 2 B, 3 中
    logits = np.array(
        [
            [0.0, 4.0, 1.0, 9.0],  # 中 is best, "a" is the best allowed class
            [5.0, 0.0, 0.0, 0.0],  # blank
            [0.0, 1.0, 9.0, 0.0],  # B
        ]
    )
    unrestricted = _ddddocr_like(tmp_path / "all", logits, charset)
    restricted = _ddddocr_like(tmp_path / "aB", logits, charset, allowed="aB", lowercase=True)
    img = np.full((40, 120, 3), 200, dtype=np.uint8)
    assert unrestricted.solve(img).text == "中B"
    assert restricted.solve(img).text == "ab"


def test_registry_loads_on_demand_and_shares_symlinks(tmp_path):
    from captcha2text.model import ModelNotFound, ModelRegistry

    models = tmp_path / "models"
    _ddddocr_like(models / "dd-v1", _path_logits([1], 2), ["a"])
    (models / "current").symlink_to("dd-v1")
    reg = ModelRegistry(models, "current")
    assert reg.default is reg.get("dd-v1")
    assert reg.default.name == "dd-test"
    for bad in ("nope", "../models/dd-v1", "dd-v1/../dd-v1"):
        with pytest.raises(ModelNotFound):
            reg.get(bad)


def test_bad_spec_is_rejected(tmp_path):
    (tmp_path / "model.json").write_text(json.dumps({"name": "x", "input": {"width": 1, "height": 1}, "decoder": "ctc"}))
    with pytest.raises(ValueError, match="charset"):
        Solver(tmp_path)
