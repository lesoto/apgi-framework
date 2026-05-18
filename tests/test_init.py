import sys
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch


def test_init_version_not_found():

    with patch("importlib.metadata.version") as mock_version:
        mock_version.side_effect = PackageNotFoundError

        apgi_backup = {}
        for k in list(sys.modules.keys()):
            if k == "apgi" or k.startswith("apgi."):
                apgi_backup[k] = sys.modules.pop(k)

        try:
            import apgi

            assert apgi.__version__ == "0.0.0+dev"
        finally:
            for k, v in apgi_backup.items():
                sys.modules[k] = v
