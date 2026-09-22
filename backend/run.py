"""Local development entry point: python -m backend.run."""

import uvicorn

uvicorn.run("backend.server.app:app", host="127.0.0.1", port=8765, reload=True)
