"""Tests for apgi.scripts.fetch_data."""

import hashlib
import pathlib
import sys
from unittest.mock import call, patch

import pytest

from apgi.scripts import fetch_data


@pytest.fixture
def mock_datasets():
    test_datasets = {
        "sim_placeholder": (
            "sim_placeholder.npz",
            "PLACEHOLDER_SHA256",
            "Placeholder test",
        ),
        "sim_sha_ok": (
            "sim_sha_ok.npz",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "Empty file test",
        ),
        "sim_sha_fail": ("sim_sha_fail.npz", "wrong_hash_here", "Wrong hash test"),
    }
    with patch.dict(fetch_data.DATASETS, test_datasets, clear=True):
        yield test_datasets


@pytest.fixture
def temp_data_dir(tmp_path):
    with patch("apgi.scripts.fetch_data.DATA_DIR", tmp_path):
        yield tmp_path


def test_verify_sha256(tmp_path):
    f_placeholder = tmp_path / "placeholder.npz"
    f_placeholder.write_bytes(b"some content")
    assert fetch_data._verify_sha256(f_placeholder, "PLACEHOLDER_SHA256") is True

    f_sha_ok = tmp_path / "sha_ok.npz"
    f_sha_ok.write_bytes(b"hello")
    hello_sha = hashlib.sha256(b"hello").hexdigest()
    assert fetch_data._verify_sha256(f_sha_ok, hello_sha) is True
    assert fetch_data._verify_sha256(f_sha_ok, "wrong_hash") is False


def test_download_already_exists_and_valid(mock_datasets, temp_data_dir):
    dest = temp_data_dir / "sim_placeholder.npz"
    dest.write_bytes(b"existing content")

    with patch("urllib.request.urlretrieve") as mock_urlretrieve:
        res = fetch_data._download("sim_placeholder")
        assert res == dest
        mock_urlretrieve.assert_not_called()


def test_download_already_exists_but_invalid(mock_datasets, temp_data_dir):
    dest = temp_data_dir / "sim_sha_ok.npz"
    dest.write_bytes(b"not empty")

    def mock_retrieve_func(url, filename):
        pathlib.Path(filename).write_bytes(b"")

    with patch(
        "urllib.request.urlretrieve", side_effect=mock_retrieve_func
    ) as mock_urlretrieve:
        res = fetch_data._download("sim_sha_ok")
        assert res == dest
        mock_urlretrieve.assert_called_once()
        assert dest.read_bytes() == b""


def test_download_not_exists_success(mock_datasets, temp_data_dir):
    dest = temp_data_dir / "sim_sha_ok.npz"
    assert not dest.exists()

    def mock_retrieve_func(url, filename):
        pathlib.Path(filename).write_bytes(b"")

    with patch(
        "urllib.request.urlretrieve", side_effect=mock_retrieve_func
    ) as mock_urlretrieve:
        res = fetch_data._download("sim_sha_ok")
        assert res == dest
        mock_urlretrieve.assert_called_once()
        assert dest.exists()


def test_download_failure(mock_datasets, temp_data_dir):
    dest = temp_data_dir / "sim_sha_ok.npz"

    def mock_retrieve_func(url, filename):
        pathlib.Path(filename).write_bytes(b"partial content")
        raise Exception("Connection refused")

    with patch("urllib.request.urlretrieve", side_effect=mock_retrieve_func):
        with pytest.raises(SystemExit) as excinfo:
            fetch_data._download("sim_sha_ok")
        assert excinfo.value.code == 1
    assert not dest.exists()


def test_script_execution_main(mock_datasets):
    import runpy

    sys_modules_backup = sys.modules.get("apgi.scripts.fetch_data")
    if "apgi.scripts.fetch_data" in sys.modules:
        del sys.modules["apgi.scripts.fetch_data"]
    try:
        with patch("sys.argv", ["fetch_data.py", "--list"]):
            runpy.run_module("apgi.scripts.fetch_data", run_name="__main__")
    finally:
        if sys_modules_backup is not None:
            sys.modules["apgi.scripts.fetch_data"] = sys_modules_backup


def test_download_post_checksum_mismatch(mock_datasets, temp_data_dir):

    def mock_retrieve_func(url, filename):
        pathlib.Path(filename).write_bytes(b"some downloaded content")

    with patch("urllib.request.urlretrieve", side_effect=mock_retrieve_func):
        with pytest.raises(SystemExit) as excinfo:
            fetch_data._download("sim_sha_fail")
        assert excinfo.value.code == 1


def test_main_list(mock_datasets, capsys):
    with patch("sys.argv", ["fetch_data.py", "--list"]):
        fetch_data.main()
    captured = capsys.readouterr()
    assert "Available datasets" in captured.out
    assert "sim_placeholder" in captured.out
    assert "sim_sha_ok" in captured.out


def test_main_dataset_valid(mock_datasets, temp_data_dir):
    with patch("sys.argv", ["fetch_data.py", "--dataset", "sim_placeholder"]):
        with patch("urllib.request.urlretrieve") as mock_urlretrieve:
            fetch_data.main()
            mock_urlretrieve.assert_called_once()


def test_main_dataset_invalid(mock_datasets, capsys):
    with patch("sys.argv", ["fetch_data.py", "--dataset", "invalid_name"]):
        with pytest.raises(SystemExit) as excinfo:
            fetch_data.main()
        assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Unknown dataset" in captured.err


def test_main_all_datasets(mock_datasets, temp_data_dir):

    def mock_retrieve_func(url, filename):
        name = pathlib.Path(filename).name
        if "sim_sha_ok" in name:
            pathlib.Path(filename).write_bytes(b"")
        else:
            pathlib.Path(filename).write_bytes(b"placeholder content")

    with patch("sys.argv", ["fetch_data.py"]):
        filtered_datasets = {
            "sim_placeholder": (
                "sim_placeholder.npz",
                "PLACEHOLDER_SHA256",
                "Placeholder test",
            ),
            "sim_sha_ok": (
                "sim_sha_ok.npz",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "Empty file test",
            ),
        }
        with patch.dict(fetch_data.DATASETS, filtered_datasets, clear=True):
            with patch(
                "urllib.request.urlretrieve", side_effect=mock_retrieve_func
            ) as mock_urlretrieve:
                fetch_data.main()
                assert mock_urlretrieve.call_count == 2


def test_download_unsafe_url_scheme(mock_datasets, temp_data_dir, capsys):
    with patch("apgi.scripts.fetch_data.ZENODO_BASE", "file:///unsafe/path"):
        with patch("sys.exit") as mock_exit:
            fetch_data._download("sim_placeholder")
            assert mock_exit.call_count >= 1
            assert call(1) in mock_exit.call_args_list
    captured = capsys.readouterr()
    assert "Unsafe URL scheme" in captured.err
