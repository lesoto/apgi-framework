import importlib
import sys
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest


def test_init_version_not_found():
    # We need to reload the module to trigger the version check
    # and we need to mock importlib.metadata.version to raise PackageNotFoundError

    with patch("importlib.metadata.version") as mock_version:
        mock_version.side_effect = PackageNotFoundError

        # If apgi is already in sys.modules, we need to remove it or reload it
        if "apgi" in sys.modules:
            importlib.reload(sys.modules["apgi"])
        else:
            import apgi

        assert sys.modules["apgi"].__version__ == "0.0.0+dev"

    # Reload again to restore original version (optional but good practice)
    importlib.reload(sys.modules["apgi"])
