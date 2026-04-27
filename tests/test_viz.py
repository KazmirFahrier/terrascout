from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from terrascout.viz.render import render_trace


class RenderTraceTest(unittest.TestCase):
    def test_render_trace_writes_png(self) -> None:
        payload = {
            "metrics": {},
            "trace": {
                "poses": [(1.0, 0.8, 0.0), (2.0, 1.2, 0.2), (3.0, 1.5, 0.3)],
                "goals": [(3.0, 1.5), (5.0, 2.0)],
                "workers": [[]],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            output_path = Path(tmp) / "trace.png"
            trace_path.write_text(json.dumps(payload))

            render_trace(trace_path, output_path, seed=7)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()

