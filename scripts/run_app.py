"""Start the SmartRoute demo server."""

from pathlib import Path
import sys

# Add the project root to Python's import path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "webapp.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )