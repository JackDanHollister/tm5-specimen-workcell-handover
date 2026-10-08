#!/usr/bin/env python3
"""Interactive offline Isaac demo and deterministic MP4 recording."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pin_axis_3d_sim.reconstructed_demo import read_plan, presentation_frames, digest
from pin_axis_3d_sim.offline_playback import OfflinePlayback

PHASES = {"spawn": "Reconstruct the next pin", "approach": "Align the jaws with the pin",
          "grasp": "Approach along the tilted shaft", "close": "Grip 5 mm below the upper endpoint",
          "extract": "Withdraw along the same pin axis", "clearance": "Clear the source drawer",
          "upright": "Turn the specimen upright", "transport": "Transfer to the destination drawer",
          "preplace": "Align above the next slot", "place": "Insert vertically into the drawer",
          "release": "Release the specimen", "retreat": "Withdraw vertically",
          "home": "Return to home", "home_pause": "Ready for the next pin"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--smoke-frames", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--film-speed", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or str(Path.home() / "miniforge3/envs/ros2-jazzy-tm/bin/ffmpeg"))
    parser.add_argument("--usd", type=Path, default=ROOT / "generated/isaac/6.0.1-watson-qc-10mm/tm5s_with_2fg7/tm5s_with_2fg7.usda")
    parser.add_argument("--import-report", type=Path, default=ROOT / "outputs/isaac_sim/6.0.1/watson_qc_10mm_import_report.json")
    args = parser.parse_args()
    if args.report.exists() or (args.video and args.video.exists()):
        raise FileExistsError("Use new paths for video and report")
    plan = read_plan(args.plan)
    asset = json.loads(args.import_report.read_text())
    if asset["output_usd_sha256"] != digest(args.usd) or asset["source_urdf_sha256"] != plan["provenance"]["urdf_sha256"]:
        raise ValueError("render asset does not match the checked planner model")
    frames = presentation_frames(plan, args.fps, args.film_speed)
    cfg = plan["scene"]["config"]
    PHASES["close"] = f"Grip {cfg['grasp_below_upper_endpoint_mm']:g} mm below the upper endpoint"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless, "width": args.width, "height": args.height,
        "renderer": "RaytracedLighting", "open_usd": str(args.usd.resolve()),
        "limit_cpu_threads": 4, "sync_loads": False, "fast_shutdown": True,
        "disable_viewport_updates": False})
    encoder = None
    report = {"status": "started", "plan_sha256": digest(args.plan), "frames": 0,
              "ros_used": False, "robot_connected": False, "physical_pick_allowed": False,
              "max_joint_tracking_error_rad": 0., "placement_checks": [], "control_checks": []}
    try:
        import omni.usd
        import omni.replicator.core as rep
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.rendering_manager import ViewportManager
        from omni.physx import get_physx_interface
        from pxr import Gf, Usd, UsdGeom, UsdLux
        from PIL import Image, ImageDraw, ImageFont

        stage = omni.usd.get_context().get_stage()
        root = str(stage.GetDefaultPrim().GetPath())
        def matrix(value):
            return Gf.Matrix4d(*np.asarray(value, dtype=float).T.reshape(-1).tolist())
        def pose(path):
            return np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        def color(prim, rgb):
            UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        def box(path, centre, size, rgb):
            item = UsdGeom.Cube.Define(stage, path)
            item.CreateSizeAttr(1.)
            item.AddTranslateOp().Set(Gf.Vec3d(*centre))
            item.AddScaleOp().Set(Gf.Vec3f(*size))
            color(item.GetPrim(), rgb)
            return item
        box("/Demo/Floor", [0., .2, -.025], [2.5, 2.5, .04], [.065, .09, .13])
        UsdLux.DomeLight.Define(stage, "/Demo/Dome").CreateIntensityAttr(1000.)
        light = UsdLux.DistantLight.Define(stage, "/Demo/Key")
        light.CreateIntensityAttr(2500.)
        light.AddRotateXYZOp().Set(Gf.Vec3f(30, -20, -20))
        for name, centre, yaw in (("Source", cfg["source_drawer_center_m"], cfg["source_drawer_yaw_deg"]),
                                   ("Destination", cfg["destination_drawer_center_m"], 0)):
            width, height = cfg["source_drawer_size_m"] if name == "Source" else cfg["drawer_size_m"]
            tray = UsdGeom.Xform.Define(stage, "/Demo/" + name)
            tray.AddTranslateOp().Set(Gf.Vec3d(*centre))
            tray.AddRotateZOp().Set(yaw)
            box(f"/Demo/{name}/Foam", [0, 0, -.02], [width, height, .04], [.80, .84, .83])
            for j, (p, s) in enumerate((([-width/2-.006, 0, .0175], [.012, height+.024, .035]),
                                        ([width/2+.006, 0, .0175], [.012, height+.024, .035]),
                                        ([0, -height/2-.006, .0175], [width, .012, .035]),
                                        ([0, height/2+.006, .0175], [width, .012, .035]))):
                box(f"/Demo/{name}/Wall{j}", p, s, [.24, .16, .10])
        colors = [(0.13,.72,.63), (.91,.57,.19), (.49,.37,.85), (.22,.58,.90), (.87,.35,.44), (.43,.72,.25), (.87,.66,.26)]
        payloads = []
        for i, (case, item) in enumerate(zip(plan["scene"]["cases"], plan["sequence"])):
            path = f"/Demo/Pin_{i+1}"
            payload = UsdGeom.Xform.Define(stage, path)
            op = payload.AddTransformOp()
            visibility = UsdGeom.Imageable(payload).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            length = case["pin_length_above_foam_m"]
            shaft = UsdGeom.Cylinder.Define(stage, path + "/Shaft")
            shaft.CreateRadiusAttr(cfg["pin_radius_m"])
            shaft.CreateHeightAttr(length + cfg["buried_pin_length_m"])
            shaft.AddTranslateOp().Set(Gf.Vec3d(0, 0, (length - cfg["buried_pin_length_m"])/2))
            color(shaft.GetPrim(), [.62,.70,.75])
            head = UsdGeom.Sphere.Define(stage, path + "/Head")
            head.CreateRadiusAttr(cfg["head_radius_m"])
            head.AddTranslateOp().Set(Gf.Vec3d(0, 0, length))
            color(head.GetPrim(), [.86,.72,.28])
            body = UsdGeom.Sphere.Define(stage, path + "/IllustrativeBody")
            body.CreateRadiusAttr(.5)
            body.AddTranslateOp().Set(Gf.Vec3d(0, 0, length * cfg["illustrative_body_fraction"]))
            body.AddScaleOp().Set(Gf.Vec3f(*cfg["illustrative_body_size_m"]))
            color(body.GetPrim(), colors[i % len(colors)])
            # The cyan extension makes the image-derived approach line legible.
            guide = UsdGeom.BasisCurves.Define(stage, path + "/AxisGuide")
            guide.CreateTypeAttr("linear")
            guide.CreateCurveVertexCountsAttr([2])
            guide.CreatePointsAttr([Gf.Vec3f(0,0,length), Gf.Vec3f(0,0,length+.08)])
            guide.CreateWidthsAttr([.00055])
            guide.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            color(guide.GetPrim(), [.1,.9,.9])
            guide_vis = UsdGeom.Imageable(guide).CreateVisibilityAttr()
            payloads.append((op, visibility, guide_vis))

        World.clear_instance()
        world = World(physics_dt=1/60, rendering_dt=1/60, stage_units_in_meters=1., backend="numpy", device="cpu")
        robot = world.scene.add(SingleArticulation(prim_path=root, name="offline_reconstructed_tm5"))
        world.reset()
        world.pause()
        if list(robot.dof_names) != [f"joint_{i}" for i in range(1, 7)]:
            raise ValueError("unexpected robot articulation")
        physx = get_physx_interface()
        tcp_path = asset["expected_link_paths"]["pin_grasp_tcp"]
        left = stage.GetPrimAtPath(asset["expected_link_paths"]["onrobot_2fg7_left_finger_link"]).GetAttribute("xformOp:translate")
        right = stage.GetPrimAtPath(asset["expected_link_paths"]["onrobot_2fg7_right_finger_link"]).GetAttribute("xformOp:translate")
        left_open, right_open = tuple(left.Get()), tuple(right.Get())
        camera_path = "/Demo/PresentationCamera"
        camera = UsdGeom.Camera.Define(stage, camera_path)
        eye, target = [1.7, -1.75, 1.35], [.20, .23, .30]
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0,0,1)).GetInverse())
        camera.CreateFocalLengthAttr(24.)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 20))
        render_product = rep.create.render_product(camera_path, (args.width, args.height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([render_product])
        detail_camera = UsdGeom.Camera.Define(stage, "/Demo/DetailCamera")
        detail_op = detail_camera.AddTransformOp()
        detail_camera.CreateFocalLengthAttr(40.)
        detail_camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 10))
        detail_product = rep.create.render_product("/Demo/DetailCamera", (360, 250))
        detail_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        detail_rgb.attach([detail_product])
        if not args.headless:
            ViewportManager.wait_for_viewport(max_frames=120)
            ViewportManager.set_camera_view(ViewportManager.get_camera(), eye=eye, target=target)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, 25)
        small = ImageFont.truetype(font_path, 17)
        title_font = ImageFont.truetype(font_path, 31)
        if args.video:
            encoder = subprocess.Popen([args.ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
                "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.video)], stdin=subprocess.PIPE)
        playback = OfflinePlayback(len(frames), speed=1.)
        requested = {"reset": False}
        if not args.headless and not args.video:
            import omni.ui as ui
            panel = ui.Window("Reconstructed Pin Demo", width=420, height=260)
            with panel.frame:
                with ui.VStack(spacing=9):
                    ui.Label("Image reconstruction → upright placement", word_wrap=True)
                    status_label = ui.Label("Ready — press Play")
                    with ui.HStack(height=34):
                        ui.Button("Play", clicked_fn=playback.play)
                        ui.Button("Pause", clicked_fn=playback.pause)
                        ui.Button("Reset", clicked_fn=lambda: requested.update(reset=True))
                    speed = ui.FloatSlider(min=.5, max=10, step=.5)
                    speed.model.set_value(1.)
                    speed.model.add_value_changed_fn(lambda m: playback.set_speed(m.as_float))
                    ui.Label(f"Grasp offset: {cfg['grasp_below_upper_endpoint_mm']:g} mm (change config and replan)", word_wrap=True)
                    ui.Label("Offline demonstration · estimated endpoints · illustrative bodies", word_wrap=True)
        else:
            status_label = None
        if args.autoplay or args.video or args.headless:
            playback.play()
        placed_poses = {}
        maximum_tcp_displacement = 0.
        initial_tcp = None
        last_index = -1
        last_time = time.monotonic()
        test_state = 0
        recorded = 0

        def reset():
            placed_poses.clear()
            playback.reset()
            report["placement_checks"].clear()
            show(0)

        def show(index):
            nonlocal initial_tcp, maximum_tcp_displacement
            frame = frames[index]
            i = frame["case"]
            sequence = plan["sequence"][i]
            robot.set_joint_positions(np.array(frame["joints"]))
            robot.set_joint_velocities(np.zeros(6))
            world.physics_sim_view.update_articulations_kinematic()
            physx.update_transformations(False, True, False)
            left.Set(Gf.Vec3d(left_open[0] - .019 * frame["grip"], *left_open[1:]))
            right.Set(Gf.Vec3d(right_open[0] + .019 * frame["grip"], *right_open[1:]))
            error = float(np.max(abs(robot.get_joint_positions() - frame["joints"])))
            report["max_joint_tracking_error_rad"] = max(report["max_joint_tracking_error_rad"], error)
            if error > .001:
                raise RuntimeError("rendered articulation tracking error")
            tcp = pose(tcp_path)
            if initial_tcp is None:
                initial_tcp = tcp[:3, 3].copy()
            maximum_tcp_displacement = max(maximum_tcp_displacement, float(np.linalg.norm(tcp[:3, 3] - initial_tcp)))
            for j, (op, visibility, guide_vis) in enumerate(payloads):
                visibility.Set(UsdGeom.Tokens.inherited if j <= i else UsdGeom.Tokens.invisible)
                guide_vis.Set(UsdGeom.Tokens.inherited if j == i and not frame["attached"] and not frame["placed"] else UsdGeom.Tokens.invisible)
                if j < i or (j == i and frame["placed"]):
                    if j not in placed_poses:
                        raise RuntimeError("placement boundary was skipped")
                    value = placed_poses[j]
                elif j == i and frame["attached"]:
                    value = tcp @ np.array(sequence["payload_in_tool"])
                    if frame["phase"] == "release":
                        placed_poses[i] = value.copy()
                else:
                    value = np.array(plan["sequence"][j]["source_payload_pose"])
                op.Set(matrix(value))
                if j == i:
                    focus = value[:3,3] + value[:3,2] * (plan["scene"]["cases"][i]["pin_length_above_foam_m"] * .7)
                    detail_eye = focus + np.array([.22, -.28, .20])
                    detail_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*detail_eye), Gf.Vec3d(*focus), Gf.Vec3d(0,0,1)).GetInverse())
            if frame["placed"] and not any(p["id"] == sequence["id"] for p in report["placement_checks"]):
                actual = placed_poses[i]
                target_entry = np.array(plan["scene"]["cases"][i]["destination_entry_m"])
                tilt = float(np.degrees(np.arccos(np.clip(actual[2,2], -1, 1))))
                error = float(np.linalg.norm(actual[:3,3] - target_entry))
                if tilt > .6 or error > .001:
                    raise RuntimeError(f"placement outside tolerance: tilt={tilt}, error={error}")
                report["placement_checks"].append({"id": sequence["id"], "tilt_deg": tilt, "entry_error_m": error})
            if status_label is not None:
                status_label.text = f"Pin {i+1}/{len(payloads)} · {PHASES[frame['phase']]}"
            world.render()
            return frame

        show(0)
        for _ in range(60):
            world.render()
        print("OFFLINE RECONSTRUCTED DEMO READY", flush=True)
        while app.is_running():
            if requested["reset"]:
                reset()
                last_index = -1
                requested["reset"] = False
            if args.video or args.headless:
                index = last_index + 1
                if index >= len(frames):
                    break
                indices = range(index, index + 1)
            else:
                now = time.monotonic()
                # Normalize the frame clock, not the user speed. This keeps
                # the full speed slider usable at any recording frame rate.
                indices = playback.advance(min(now - last_time, .1) * args.fps / 60)
                last_time = now
            frame = frames[max(0, last_index)]
            for index in indices:
                frame = show(index)
                last_index = index
                report["frames"] += 1
            if not indices:
                world.render()
            if args.self_test and test_state == 0 and frame["attached"]:
                playback.pause()
                before = pose(tcp_path)
                for _ in range(5):
                    world.render()
                if not np.allclose(before, pose(tcp_path), atol=1e-8):
                    raise RuntimeError("pause did not hold carried pin")
                report["control_checks"].append("pause_during_carry")
                reset()
                if placed_poses or report["placement_checks"] or playback.state != "ready":
                    raise RuntimeError("reset did not clear playback and placements")
                if not np.allclose(robot.get_joint_positions(), cfg["ready_joints_rad"], atol=1e-6):
                    raise RuntimeError("reset did not return the arm home")
                for j, (op, visibility, _) in enumerate(payloads):
                    expected_visibility = UsdGeom.Tokens.inherited if j == 0 else UsdGeom.Tokens.invisible
                    if visibility.Get() != expected_visibility:
                        raise RuntimeError("reset did not restore one source pin")
                    if not np.allclose(np.array(op.Get()).T, plan["sequence"][j]["source_payload_pose"]):
                        raise RuntimeError("reset did not restore source pin poses")
                if tuple(left.Get()) != left_open or tuple(right.Get()) != right_open:
                    raise RuntimeError("reset did not open the fingers")
                report["control_checks"].append("reset_during_carry")
                test_state = 1
                last_index = -1
                playback.play()
                continue
            if encoder and indices:
                # Annotators need an explicit capture step while the kinematic
                # timeline is paused. Zero delta time preserves the checked pose.
                rep.orchestrator.step(delta_time=0.0, rt_subframes=1, pause_timeline=True)
                pixels = np.asarray(rgb.get_data())
                if pixels.size == 0:
                    raise RuntimeError("RGB render product returned no frame")
                picture = Image.fromarray(pixels[:,:,:3].astype(np.uint8))
                detail_pixels = np.asarray(detail_rgb.get_data())
                inset_x, inset_y = args.width - 384, args.height - 380
                if detail_pixels.size:
                    picture.paste(Image.fromarray(detail_pixels[:,:,:3].astype(np.uint8)), (inset_x, inset_y))
                draw = ImageDraw.Draw(picture)
                draw.rectangle([inset_x-1, inset_y-1, inset_x+361, inset_y+251], outline=(65,198,185), width=2)
                draw.rectangle([inset_x, inset_y-26, inset_x+360, inset_y], fill=(13,23,34))
                draw.text((inset_x+10,inset_y-23), "PIN / GRIPPER DETAIL", font=small, fill=(105,222,205))
                draw.rectangle([0,0,args.width,91], fill=(13,23,34))
                draw.text((26,15), "From images to an upright specimen drawer", font=title_font, fill=(235,243,249))
                draw.text((27,57), f"TM5 digital twin  /  {len(payloads)} reconstructed pin placements  /  offline demonstration", font=small, fill=(158,191,207))
                draw.rectangle([0,args.height-106,args.width,args.height], fill=(13,23,34))
                i = frame["case"]
                case = plan["scene"]["cases"][i]
                caption = PHASES[frame["phase"]]
                if frame["phase"] == "close":
                    caption = f"Grip {cfg['grasp_below_upper_endpoint_mm']:g} mm below the upper endpoint"
                draw.text((26,args.height-94), f"PIN {i+1} / {len(payloads)}   {caption}", font=font, fill=(235,243,249))
                draw.text((27,args.height-55), f"Image-derived axis: {case['tilt_deg']:.1f}° tilt    |    Destination: {i + int(frame['placed'])} placed", font=small, fill=(105,222,205))
                draw.text((27,args.height-29), "Image-derived axes · provisional shaft endpoints · illustrative specimen bodies · no physical robot", font=small, fill=(158,175,189))
                encoder.stdin.write(picture.tobytes())
                recorded += 1
                if recorded in {1, args.fps * 4, args.fps * 8}:
                    picture.save(args.video.with_name(f"preview-{recorded}.png"))
            if args.smoke_frames and report["frames"] >= args.smoke_frames:
                break
            if last_index >= 0 and last_index % 60 == 0:
                print(f"Frame {last_index}/{len(frames)} pin {frame['case']+1} {frame['phase']}", flush=True)
                args.report.with_suffix(".progress.json").write_text(json.dumps({
                    "frame": last_index, "total_frames": len(frames), "pin": frame["case"]+1,
                    "phase": frame["phase"], "status": "rendering"}) + "\n")
        report["max_rendered_tcp_displacement_m"] = maximum_tcp_displacement
        report["completed_transfers"] = len(report["placement_checks"])
        if args.smoke_frames:
            report["status"] = "smoke_passed"
        elif last_index == len(frames) - 1 and report["completed_transfers"] == len(payloads) and maximum_tcp_displacement > .1:
            report["status"] = "passed"
        else:
            report["status"] = "incomplete"
        report["video_frames"] = recorded
        if encoder:
            encoder.stdin.close()
            if encoder.wait(timeout=60) != 0:
                raise RuntimeError("video encoder failed")
            encoder = None
            report["video_sha256"] = digest(args.video)
        print(json.dumps(report), flush=True)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        if encoder:
            encoder.stdin.close()
            encoder.wait(timeout=60)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        app.close()


if __name__ == "__main__":
    main()
