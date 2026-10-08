#!/usr/bin/env python3
"""Verify every distributed robot asset and the frozen demonstration plan."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(bundle: Path, manifest: dict) -> None:
    bundle = bundle.resolve()
    for name, expected in manifest["files"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or not path.is_file():
            raise ValueError(f"Missing or unsafe asset: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Asset hash mismatch: {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("release_bundle"))
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "assets-manifest.json").read_text())
    verify(args.bundle, manifest)
    print(f"Verified {len(manifest['files'])} demo assets")
