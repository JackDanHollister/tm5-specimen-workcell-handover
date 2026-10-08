#!/usr/bin/env python3
"""Package only manifest-listed assets and successful demonstration evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from verify_demo_assets import verify


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(root: Path, bundle: Path, controls: Path, reset: Path, output: Path) -> None:
    manifest = json.loads((root / "assets-manifest.json").read_text())
    verify(bundle, manifest)
    video = bundle / "evidence/tm5-reconstructed-pin-demo.mp4"
    video_report = bundle / "evidence/video-validation.json"
    expected_plan = manifest["files"]["evidence/transfer-plan.json"]
    for path in (video_report, controls):
        report = json.loads(path.read_text())
        if report["status"] != "passed" or report["completed_transfers"] != 7:
            raise ValueError(f"Incomplete validation: {path}")
        if report["plan_sha256"] != expected_plan:
            raise ValueError(f"Validation is for a different plan: {path}")
        if report["ros_used"] or report["robot_connected"] or report["physical_pick_allowed"]:
            raise ValueError(f"Unexpected hardware authority: {path}")
    recording = json.loads(video_report.read_text())
    if recording["video_sha256"] != sha256(video):
        raise ValueError("Video does not match its validation report")
    control_result = json.loads(controls.read_text())
    if not {"pause_during_carry", "reset_during_carry"}.issubset(control_result["control_checks"]):
        raise ValueError("Missing playback control checks")
    reset_result = json.loads(reset.read_text())
    if (reset_result["status"] != "smoke_passed" or
            reset_result["plan_sha256"] != expected_plan or
            "reset_during_carry" not in reset_result["control_checks"]):
        raise ValueError("Missing final reset regression check")
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "demo-assets.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for relative in sorted(manifest["files"]):
            zipped.write(bundle / relative, "release_bundle/" + relative)
    for source, name in ((video, video.name), (video_report, video_report.name),
                         (controls, "controls-validation.json"),
                         (reset, "reset-regression.json"),
                         (root / "assets-manifest.json", "assets-manifest.json")):
        shutil.copy2(source, output / name)
    files = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files))
    print(f"Packaged {len(files)} release assets in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("release_bundle"))
    parser.add_argument("--controls-report", type=Path, required=True)
    parser.add_argument("--reset-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(Path(__file__).resolve().parents[1], args.bundle, args.controls_report, args.reset_report, args.output)
