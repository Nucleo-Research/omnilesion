"""Container entry point: segment every CT in /workspace/inputs and write uint8 lesion masks to /workspace/outputs.

The whole pipeline lives in the ``omnilesion`` package copied into the image; this file only fixes the harness paths.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")
from omnilesion import inference  # noqa: E402

INPUTS = Path(os.environ.get("FLARE_INPUTS", "/workspace/inputs"))
OUTPUTS = Path(os.environ.get("FLARE_OUTPUTS", "/workspace/outputs"))

if __name__ == "__main__":
    sys.exit(inference.run(INPUTS, OUTPUTS, Path(os.environ.get("OMNILESION_MODEL", "/opt/model"))))
