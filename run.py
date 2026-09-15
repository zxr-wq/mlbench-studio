"""Single-command development and production entry point."""

from __future__ import annotations

import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("ML_STUDIO_PORT", "8765"))
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=False)
