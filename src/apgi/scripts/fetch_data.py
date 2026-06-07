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
# Registry: name → (filename, sha256, description, approx_size_mb)
DATASETS: dict[str, tuple[str, str, str, str]] = {
    "sim1_ignition_dynamics": (
        "sim1_ignition_dynamics.npz",
        "213c3aad7c9c5f8198f30f985c470cfd7dd647e2cd78c0783fc18ad53b5d0b9e",
        "Ignition dynamics simulation for Figure 1 (n=10 000 trials)",
        "~1.5 MB",
    ),
    "sim2_parameter_recovery": (
        "sim2_parameter_recovery.npz",
        "279c81899e413d648a657c679157108cb56de441004d8d8ad7d9c6fa0279eb5a",
        "Parameter recovery simulation results for Figure 2 (n=1 000 runs)",
        "~0.1 MB",
    ),
    "sim3_liquid_network": (
        "sim3_liquid_network.npz",
        "11fd84ada6e39004db536e12dcc6718df071bb744f7898859236a5e2808275d5",
        "LNN reservoir state trajectories — Paper 2 supplementary (~40 MB, largest dataset)",
        "~40 MB",
    ),
    "sim4_hierarchical": (
        "sim4_hierarchical.npz",
        "50c10976ace922726e23cf27cdb60e47b96c0b3a5e0c6c2a8b6cab37d79bc7cf",
        "Five-level hierarchy prediction-error series — Paper 3",
        "~2 MB",
    ),
    "sim5_doc_biomarker": (
        "sim5_doc_biomarker.npz",
        "8b1e69f3b58c760ca8c7a76b9d974eb716435fef38689aca39a78489f235f49c",
        "DoC biomarker simulation (VS/UWS, MCS, Controls) — Figure 6, Protocol 6",
        "~1 MB",
    ),
    "sim6_bifurcation": (
        "sim6_bifurcation.npz",
        "4eefebf97fb01870178e53a2f6bb43a890ce75aa36a232cc55ba37e6cd943aec",
        "LNN bifurcation CSD signatures — Figure 7, bifurcation analysis script",
        "~0.05 MB",
    ),
}


def _verify_sha256(path: pathlib.Path, expected: str) -> bool:
    if expected == "PLACEHOLDER_SHA256":
        return True  # skip verification until real DOI is set
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest == expected


def _verify_cached(names: list[str] | None = None) -> bool:
    """Check SHA-256 of all (or selected) cached files without re-downloading."""
    targets = names if names else list(DATASETS.keys())
    all_ok = True
    for name in targets:
        filename, sha256, _, _ = DATASETS[name]
        path = DATA_DIR / filename
        if not path.exists():
            print(f"[missing] {name}")
            all_ok = False
            continue
        ok = _verify_sha256(path, sha256)
        status = "ok     " if ok else "MISMATCH"
        print(f"[{status}] {name}  {path}")
        if not ok:
            all_ok = False
    return all_ok


def _download(name: str) -> pathlib.Path:
    filename, sha256, _, size_hint = DATASETS[name]
    url = f"{ZENODO_BASE}/{filename}"
    dest = DATA_DIR / filename

    if dest.exists():
        if _verify_sha256(dest, sha256):
            print(f"[cache] {name} — already downloaded.")
            return dest
        print(f"[warn] {name} — checksum mismatch, re-downloading.")
        dest.unlink()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {name}  {size_hint}  — {url}")

    # Validate URL scheme to prevent file:// or other unsafe schemes
    if not url.startswith(("http://", "https://")):
        print(f"[error] Unsafe URL scheme: {url}", file=sys.stderr)
        sys.exit(1)

    try:
        urllib.request.urlretrieve(url, dest)  # nosec: B310
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
        "--dataset", metavar="NAME", help="Target a specific dataset by name"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify SHA-256 of cached files without re-downloading. Exit 1 on mismatch.",
    )
    args = parser.parse_args()

    if args.list:
        print(f"Available datasets (Zenodo DOI: {ZENODO_DOI}):\n")
        for name, (filename, _, description, size) in DATASETS.items():
            print(f"  {name:<35} {size:<10}  {filename}")
            print(f"      {description}")
        return

    if args.verify:
        targets = [args.dataset] if args.dataset else None
        ok = _verify_cached(targets)
        sys.exit(0 if ok else 1)

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
