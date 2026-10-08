"""Audit Isaac/robot frames and reconstruct saved specimen groups entirely offline."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from frames import handeye_transform, intrinsics_for, saved_flange, urdf_link_poses
from geometry import extract_axis
from inference import sha256
from multiview import observation, solve

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent.parent
DEFAULT_CALIBRATION = Path(__file__).resolve().parents[2] / 'data/pins/historical_eih_metadata.json'
DEFAULT_URDF = Path(__file__).resolve().parents[2] / 'assets/robots/no_scanner_gripper/robot.urdf'

def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--fixed-placements-confirmed", action="store_true",
                        help="Use only after the operator confirms fixed specimens within folders")
    args = parser.parse_args()
    if not args.fixed_placements_confirmed:
        parser.error("Fixed specimen placement within each folder must be confirmed")
    root = args.run.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    calibration = json.loads(args.calibration.read_text())["camera"]
    T_fc = handeye_transform(calibration)
    records, fk_errors = [], []
    state = {"status": "running", "stage": "image geometry", "completed": 0, "total": len(manifest)}
    json_write(root / "reconstruction-state.json", state)
    try:
        for row in manifest:
            folder = root / "cache" / row["id"]
            meta = json.loads((folder / "input.json").read_text())
            image_path = Path(row["path"])
            capture = json.loads(Path(row["coordinates_path"]).read_text())
            if sha256(image_path) != meta["sha256"] or capture["image"]["sha256"] != meta["sha256"]:
                raise ValueError(f"Image hash mismatch: {row['id']}")
            image = Image.open(image_path).convert("RGB")
            probs = np.load(folder / "probabilities.npz", allow_pickle=False)["probabilities"]
            result = extract_axis(np.array(image), probs)
            json_write(folder / "geometry.json", result)
            coord = capture["coordinates"]
            flange = saved_flange(coord)
            joints = dict(zip(coord["joint_names"], coord["joint_rad"]))
            links = urdf_link_poses(args.urdf, joints)
            fk = links["flange"]
            delta = np.linalg.inv(flange) @ fk
            fk_errors.append({"id": row["id"], "translation_mm": float(np.linalg.norm(delta[:3, 3]) * 1000),
                              "rotation_deg": float(np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude()))})
            K, distortion = intrinsics_for(calibration, capture["camera_settings"]["focus"], image.size)
            camera = flange @ T_fc
            records.append({**row, "source_sha256": meta["sha256"], "geometry": result,
                            "capture_accepted": capture["accepted"], "focus": capture["camera_settings"]["focus"],
                            "source_size": list(image.size), "joint_rad": coord["joint_rad"],
                            "coordinates_sha256": sha256(Path(row["coordinates_path"])),
                            "flange_pose": flange.tolist(), "camera_pose": camera.tolist(),
                            "K": K.tolist(), "distortion": distortion.tolist(),
                            "robot_link_positions": {name: links[name][:3, 3].tolist() for name in
                                                     ["base", "link_0", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6", "flange"]}})
            state["completed"] += 1
            if state["completed"] % 10 == 0:
                json_write(root / "reconstruction-state.json", state)
                print("Geometry", state["completed"], "/", len(manifest), flush=True)
        json_write(root / "detections.json", records)
        state["stage"] = "multiview fits"
        json_write(root / "reconstruction-state.json", state)
        groups = []
        for specimen in sorted({r["specimen_id"] for r in records}):
            views = [r for r in records if r["specimen_id"] == specimen]
            usable = [r for r in views if r["capture_accepted"] and r["geometry"]["angle_deg"] is not None]
            hypotheses = {}
            for name in ("factory_camera_to_flange", "inverse_handeye_diagnostic", "flange_as_camera_control"):
                obs = []
                for r in usable:
                    flange = np.array(r["flange_pose"])
                    pose = flange @ T_fc if name == "factory_camera_to_flange" else (
                        flange @ np.linalg.inv(T_fc) if name == "inverse_handeye_diagnostic" else flange)
                    obs.append(observation(r["id"], r["geometry"]["endpoints_xy"], pose,
                                           np.array(r["K"]), np.array(r["distortion"])))
                hypotheses[name] = solve(obs)
            item = {"specimen_id": specimen, "image_count": len(views), "detected_view_count": len(usable),
                    "focus_counts": dict(Counter(str(r["focus"]) for r in views)),
                    "fixed_placement": "confirmed by operator", "primary": hypotheses["factory_camera_to_flange"],
                    "transform_diagnostics": hypotheses}
            groups.append(item)
            p = item["primary"]
            print("Specimen", specimen, p["status"], "detected", len(usable), "inliers",len(p["inlier_ids"]),
                  "tilt",p.get("tilt_from_base_z_deg"), "residual",p.get("median_inlier_residual_px"),flush=True)
        audit = {"urdf_path": str(args.urdf.resolve()), "urdf_sha256": sha256(args.urdf),
                 "rotation_convention": "Rz(yaw) Ry(pitch) Rx(roll); original controller flange poses in robot base",
                 "simulation_world_transform": "base coincident with robot base; no virtual drawer offsets applied",
                 "fk_comparison": fk_errors,
                 "median_fk_translation_error_mm": float(np.median([r["translation_mm"] for r in fk_errors])),
                 "max_fk_translation_error_mm": max(r["translation_mm"] for r in fk_errors),
                 "median_fk_rotation_error_deg": float(np.median([r["rotation_deg"] for r in fk_errors])),
                 "max_fk_rotation_error_deg": max(r["rotation_deg"] for r in fk_errors),
                 "camera_calibration_source": str(args.calibration.resolve()),
                 "camera_calibration_sha256": sha256(args.calibration), "T_flange_camera": T_fc.tolist(),
                 "camera_calibration_status": "historical factory values; direction not independently physically verified",
                 "camera_settings_match": "Exact saved focus and pixel dimensions required; no approximate K scaling",
                 "fixed_placements_confirmed": True, "robot_network_used": False,
                 "source_code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}}
        json_write(root / "frame-audit.json", audit)
        json_write(root / "reconstruction.json", {"frame_audit": audit, "specimens": groups})
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        json_write(root / "reconstruction-state.json", state)


if __name__ == "__main__":
    main()
