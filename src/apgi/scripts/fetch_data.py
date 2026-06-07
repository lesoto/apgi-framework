"""Download pre-computed APGI simulation data from Zenodo.

Usage:
    python data/fetch_data.py                  # download all datasets
    python data/fetch_data.py --list           # list available datasets
    python data/fetch_data.py --dataset sim1   # download specific dataset
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import urllib.request

# Zenodo DOI: update this to the real DOI after creating the Zenodo release.
ZENODO_DOI = "10.5281/zenodo.XXXXXXX"
ZENODO_BASE = "https://zenodo.org/record/XXXXXXX/files"

DATA_DIR = pathlib.Path(__file__).parent / "cache"

# Registry: name → (filename, sha256, description)
DATASETS: dict[str, tuple[str, str, str]] = {
    "sim1_ignition_dynamics": (
        "sim1_ignition_dynamics.npz",
        "PLACEHOLDER_SHA256",
        "Ignition dynamics simulation for Figure 1 (n=10 000 trials)",
    ),
    "sim2_parameter_recovery": (
        "sim2_parameter_recovery.npz",
        "PLACEHOLDER_SHA256",
        "Parameter recovery simulation results for Figure 2 (n=1 000 runs)",
    ),
    "sim3_liquid_network": (
        "sim3_liquid_network.npz",
        "PLACEHOLDER_SHA256",
        "LNN reservoir state trajectories — Paper 2 supplementary",
    ),
    "sim4_hierarchical": (
        "sim4_hierarchical.npz",
        "PLACEHOLDER_SHA256",
        "Five-level hierarchy prediction-error series — Paper 3",
    ),
}


def _verify_sha256(path: pathlib.Path, expected: str) -> bool:
    if expected == "PLACEHOLDER_SHA256":
        return True  # skip verification until real DOI is set
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest == expected


def _download(name: str) -> pathlib.Path:
    filename, sha256, _ = DATASETS[name]
    url = f"{ZENODO_BASE}/{filename}"
    dest = DATA_DIR / filename

    if dest.exists():
        if _verify_sha256(dest, sha256):
            print(f"[cache] {name} — already downloaded.")
            return dest
        print(f"[warn] {name} — checksum mismatch, re-downloading.")
        dest.unlink()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {name} — {url}")

    # Validate URL scheme to prevent file:// or other unsafe schemes
    if not url.startswith(("http://", "https://")):
        print(f"[error] Unsafe URL scheme: {url}", file=sys.stderr)
        sys.exit(1)

    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        print(f"[error] Could not download {name}: {exc}", file=sys.stderr)
        print(
            f"  Manual download: https://doi.org/{ZENODO_DOI}",
            file=sys.stderr,
        )
        if dest.exists():
            dest.unlink()
        sys.exit(1)

    if not _verify_sha256(dest, sha256):
        print(
            f"[error] SHA-256 mismatch for {name}. File may be corrupted.",
            file=sys.stderr,
        )
        dest.unlink()
        sys.exit(1)

    print(f"[ok]    {name} → {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch APGI simulation data from Zenodo"
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument(
        "--dataset", metavar="NAME", help="Download a specific dataset by name"
    )
    args = parser.parse_args()

    if args.list:
        print(f"Available datasets (Zenodo DOI: {ZENODO_DOI}):\n")
        for name, (filename, _, description) in DATASETS.items():
            print(f"  {name:<35} {filename}")
            print(f"      {description}")
        return

    if args.dataset:
        if args.dataset not in DATASETS:
            print(
                f"Unknown dataset: {args.dataset!r}. Use --list to see options.",
                file=sys.stderr,
            )
            sys.exit(1)
        _download(args.dataset)
    else:
        for name in DATASETS:
            _download(name)


if __name__ == "__main__":
    main()
