"""Coverage shim for scripts/fetch_data.py.

The file is a two-line delegating shim to apgi.scripts.fetch_data.main.
The delegated logic is already fully tested in test_fetch_data.py.
This test ensures the shim itself is measured by the coverage report.
"""

import importlib.util
import sys
from pathlib import Path

_SHIM = Path(__file__).parent.parent / "scripts" / "fetch_data.py"


def test_shim_imports_cleanly():
    spec = importlib.util.spec_from_file_location("scripts_fetch_data_shim", _SHIM)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scripts_fetch_data_shim"] = mod
    spec.loader.exec_module(mod)
    # The shim re-exports `main` from the installed package
    assert callable(mod.main)
