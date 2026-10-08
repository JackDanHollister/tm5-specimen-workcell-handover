"""Create a static robot-base USD overlay with no physics or control components.

Run with the existing Isaac Python environment for pxr; SimulationApp is not
imported and Isaac/ROS/robot services are not started.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt


def curve(stage, path, points, color, width=0.001):
    prim = UsdGeom.BasisCurves.Define(stage, path)
    prim.CreateTypeAttr("linear")
    prim.CreateWrapAttr("nonperiodic")
    prim.CreateCurveVertexCountsAttr([len(points)])
    prim.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in points]))
    prim.CreateWidthsAttr([width])
    prim.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return prim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    results = json.loads((root / "reconstruction.json").read_text())
    views = json.loads((root / "detections.json").read_text())
    output = root / "tm5-pin-reconstruction.usda"
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    base = UsdGeom.Xform.Define(stage, "/TM5PinReconstruction")
    stage.SetDefaultPrim(base.GetPrim())
    base.GetPrim().CreateAttribute("coordinateFrame", Sdf.ValueTypeNames.String).Set("robot_base")
    base.GetPrim().CreateAttribute("validationStatus", Sdf.ValueTypeNames.String).Set(
        "Provisional factory-calibrated line fits. Not independently measured absolute accuracy.")
    base.GetPrim().CreateAttribute("displayLengthNotice", Sdf.ValueTypeNames.String).Set(
        "All cylinders are 50 mm illustrative axis segments; physical head, tip and length are unmeasured.")
    colors = [(0.2,.84,.72),(1,.65,.36),(.42,.73,1),(.9,.54,1),(1,.44,.51),(.77,.87,.39),(.66,.62,1)]
    for index, group in enumerate(results["specimens"]):
        solution = group["primary"]
        if solution["status"] != "provisional_consensus":
            continue
        path = f"/TM5PinReconstruction/Specimen_{index + 1}"
        holder = UsdGeom.Xform.Define(stage, path)
        holder.GetPrim().CreateAttribute("specimenId", Sdf.ValueTypeNames.String).Set(group["specimen_id"])
        holder.GetPrim().CreateAttribute("axisRobotBase", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*solution["axis_base"]))
        holder.GetPrim().CreateAttribute("tiltFromBaseZDegrees", Sdf.ValueTypeNames.Double).Set(solution["tilt_from_base_z_deg"])
        holder.GetPrim().CreateAttribute("supportingViews", Sdf.ValueTypeNames.Int).Set(len(solution["inlier_ids"]))
        cylinder = UsdGeom.Cylinder.Define(stage, path + "/ProvisionalAxis")
        cylinder.CreateAxisAttr("Z")
        cylinder.CreateHeightAttr(.05)
        cylinder.CreateRadiusAttr(.0004)
        cylinder.CreateDisplayColorAttr([Gf.Vec3f(*colors[index % len(colors)])])
        xf = UsdGeom.Xformable(cylinder)
        xf.AddTranslateOp().Set(Gf.Vec3d(*solution["display_center_base_m"]))
        rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*solution["axis_base"]))
        xf.AddOrientOp().Set(Gf.Quatf(rotation.GetQuat()))
        camera_group = UsdGeom.Xform.Define(stage, path + "/SavedCameraFrames")
        UsdGeom.Imageable(camera_group).CreateVisibilityAttr("invisible")
        for row in views:
            if row["specimen_id"] != group["specimen_id"]:
                continue
            pose = np.array(row["camera_pose"])
            center = pose[:3, 3]
            for axis, color in enumerate(((1,.2,.2),(.2,1,.3),(.3,.5,1))):
                curve(stage, f"{path}/SavedCameraFrames/{row['id']}_{axis}",
                      [center, center + pose[:3, axis] * .012], color, .0007)
    for axis, color in enumerate(((1,.2,.2),(.2,1,.3),(.3,.5,1))):
        end = np.eye(3)[axis] * .1
        curve(stage, f"/TM5PinReconstruction/BaseAxis_{axis}", [[0,0,0], end], color, .002)
    stage.GetRootLayer().Save()
    # Re-open the actual artifact, inspect its geometry and frame metadata.
    checked = Usd.Stage.Open(str(output))
    cylinders = [prim for prim in checked.Traverse() if prim.IsA(UsdGeom.Cylinder)]
    expected = sum(g["primary"]["status"] == "provisional_consensus" for g in results["specimens"])
    assert len(cylinders) == expected
    assert UsdGeom.GetStageMetersPerUnit(checked) == 1.0
    assert UsdGeom.GetStageUpAxis(checked) == "Z"
    for prim in checked.Traverse():
        assert not any("Physics" in schema for schema in prim.GetAppliedSchemas())
    for prim in cylinders:
        parent = prim.GetParent()
        intended = np.array(parent.GetAttribute("axisRobotBase").Get())
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        actual = np.array(matrix.TransformDir(Gf.Vec3d(0,0,1)))
        actual /= np.linalg.norm(actual)
        assert abs(actual @ intended) > 0.999999
    (root / "usd-validation.json").write_text(json.dumps({"file": str(output), "cylinders": len(cylinders),
         "meters_per_unit": 1, "up_axis": "Z", "all_axis_directions_match": True,
         "physics_or_robot_control": False}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
