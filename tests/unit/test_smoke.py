import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import industrial_safety  # noqa: E402


def test_package_imports():
    assert industrial_safety is not None
