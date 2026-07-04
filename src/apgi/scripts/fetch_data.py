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

# Registry: name → (filename, sha256, description, approx_size_mb)
DATASETS: dict[str, tuple[str, str, str, str]] = {
    "sim0_hep_proxy": (
        "sim0_hep_proxy.npz",
        "160d19715985eea2b488e145a10be5de44a62a1ff2fcecb1877285571d430ead",
        "HEP proxy validation (behavioral, pharmacological, fMRI-EEG) — Figure 0, Protocol 0",
        "~0.1 MB",
    ),
    "sim1_ignition_dynamics": (
        "sim1_ignition_dynamics.npz",
        "975c608534c72424bab023f64a2bf0205379fa3a76a401e853766b632dad7946",
        "Ignition dynamics simulation for Figure 1 (n=10 000 trials)",
        "~1.5 MB",
    ),
    "sim2_parameter_recovery": (
        "sim2_parameter_recovery.npz",
        "bca75d35b6445bfc2058fda562cf6313e8ce9b2a1ff0d5dbdb02479a9dfb53bb",
        "Parameter recovery simulation results for Figure 2 (n=1 000 runs)",
        "~0.1 MB",
    ),
    "sim3_liquid_network": (
        "sim3_liquid_network.npz",
        "b61cc629f35ad83edf2f1c4aff165fef8b30814461bdb64e5be67d5e6d35a1b1",
        "LNN reservoir state trajectories — Paper 2 supplementary (~8 MB, largest dataset)",
        "~8 MB",
    ),
    "sim4_hierarchical": (
        "sim4_hierarchical.npz",
        "fc9ec5decb337dcf190fa164ced6a8aaaf4523b1c7f2f6d2d2409ac1cb0c23aa",
        "Five-level hierarchy prediction-error series — Figure 4 (Paper 1)",
        "~0.5 MB",
    ),
    "sim5_doc_biomarker": (
        "sim5_doc_biomarker.npz",
        "bd8bfca71010781ff08dd96e278187b53fc76400f42598cc8c84d3c8f365a56d",
        "DoC biomarker simulation (VS/UWS, MCS, EMCS, Controls) — Figure 7, Protocol 7",
        "~0.2 MB",
    ),
    "sim6_bifurcation": (
        "sim6_bifurcation.npz",
        "9aa0badbfd58ef8f760105115f76c8ce613afb17313a54f70df8efbe78c0e293",
        "LNN bifurcation CSD signatures — Figure 6, bifurcation analysis script",
        "~0.05 MB",
    ),
    "sim7_metabolic_crossover": (
        "sim7_metabolic_crossover.npz",
        "654b7bd8a6cc5b48af14f82d6c5d7511aaf6c2805797d40ed502b6bca9df6759",
        "2x2 metabolic crossover simulation — Figure 4b, Protocol 4",
        "~0.02 MB",
    ),
    "sim8_tms_pci": (
        "sim8_tms_pci.npz",
        "b235616778670083cc4c78e18cf969132ce6d221a63921ac823b40af7cbe8012",
        "TMS/tFUS site comparison (aINS/dlPFC/vertex) PCI+HEP — Figure 5, Protocol 5",
        "~0.01 MB",
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
