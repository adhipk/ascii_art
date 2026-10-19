"""Execute the worker's actual Python expression for both edge-presence states."""
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "worker_boundary_bridge", ROOT / "web/bridge.py"
)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class JsNull:
    """Reproduce modern Pyodide's non-None representation of JS null."""

    pass


class BufferProxy:
    def __init__(self, buffer):
        self.buffer = buffer

    def to_py(self):
        return memoryview(self.buffer)


@pytest.mark.parametrize("has_edges", [False, True])
def test_worker_expression_renders_text_with_and_without_edge_buffer(has_edges):
    source = (ROOT / "web/python-render.ts").read_text()
    code = re.search(r"export const PYTHON_RENDER_CODE = `([\s\S]*?)`;", source).group(
        1
    )
    rgba = np.full((32, 32, 4), 255, np.uint8)
    namespace = {
        "render_rgba": bridge.render_rgba,
        "json": json,
        "_pixels": BufferProxy(rgba),
        "_width": 32,
        "_height": 32,
        "_settings": json.dumps(
            {
                "columns": 20,
                "use_edge": has_edges,
                "edge_method": "tiny",
                "edge_amount": 20,
                "edge_sensitivity": 50,
                "text_metrics": {
                    "width": 8,
                    "height": 17,
                    "densities": [
                        0,
                        0.01,
                        0.03,
                        0.06,
                        0.09,
                        0.12,
                        0.16,
                        0.20,
                        0.25,
                        0.30,
                    ],
                },
            }
        ),
        "_has_edges": has_edges,
        "_edge_pixels": BufferProxy(np.zeros((16, 16), np.float32))
        if has_edges
        else JsNull(),
        "_edge_width": 16 if has_edges else 0,
        "_edge_height": 16 if has_edges else 0,
    }
    result = eval(code, namespace)
    assert result["columns"] == 20
    assert "png" not in result
    assert set(result["text"]) <= {" ", "\n"}
