from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    try:
        import fastapi  # noqa: F401
        import uvicorn
    except Exception:
        print("UI dependencies missing. Install with:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    print("Starting UI at http://127.0.0.1:8000")
    uvicorn.run("app.web:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
