"""Open a packaged drawer/model and replay recorded joint poses entirely offline."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from workcell_kit.kinematics import fk


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drawer',default='drawer_001')
    parser.add_argument('--robot',default='scanner_no_gripper')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--replay',action='store_true')
    parser.add_argument('--frames',type=int,default=180)
    parser.add_argument('--screenshot',type=Path)
    parser.add_argument('--report',type=Path,default=ROOT/'outputs/isaac-report.json')
    args=parser.parse_args()
    models=json.loads((ROOT/'configs/robots.json').read_text())
    model=next(x for x in models if x['name']==args.robot)
    drawers=json.loads((ROOT/'configs/drawers.json').read_text())
    drawer=next(x for x in drawers if x['id']==args.drawer)
    scene=ROOT/'assets/scenes'/(args.drawer+'__'+args.robot+'.usda')
    if not scene.exists():raise FileNotFoundError('Download the private release assets first')
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':args.headless,'width':1440,'height':1080,'renderer':'RaytracedLighting',
        'open_usd':str(scene),'limit_cpu_threads':4,'sync_loads':False,'fast_shutdown':True,
        'multi_gpu':False,'max_gpu_count':1})
    report={'status':'started','drawer':args.drawer,'robot':args.robot,'robot_connected':False,
            'physical_pick_allowed':False,'maximum_fk_error_m':0.,'frames':0}
    try:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from omni.physx import get_physx_interface
        from pxr import Usd,UsdGeom
        stage=omni.usd.get_context().get_stage()
        World.clear_instance()
        world=World(physics_dt=1/60,rendering_dt=1/60,stage_units_in_meters=1.,backend='numpy',device='cpu')
        robot=world.scene.add(SingleArticulation(prim_path='/World/Robot',name='offline_workcell_robot'))
        world.reset();world.pause()
        indices=np.array([list(robot.dof_names).index(f'joint_{i}') for i in range(1,7)])
        physx=get_physx_interface()
        base_root=model['link_paths']['base'].split('/Geometry/')[0]
        scene_paths={k:p.replace(base_root,'/World/Robot',1) for k,p in model['link_paths'].items()}
        scan_prim=stage.GetPrimAtPath('/World/Drawer/MeasuredSurface')
        scan_before=np.asarray(UsdGeom.Xformable(scan_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).copy()
        states=[np.asarray(drawer['home_joints_rad'])]
        if args.self_test:
            states.extend([states[0]+np.array([.08,0,0,0,0,0]),states[0]+np.array([0,.04,-.04,0,0,0])])
        if args.replay:
            route=json.loads((ROOT/drawer['folder']/'recorded_route.json').read_text())
            states.extend(np.asarray(q) for move in route for q in move['joint_samples_rad'])
        for index,q in enumerate(states):
            robot.set_joint_positions(q,joint_indices=indices)
            robot.set_joint_velocities(np.zeros(len(robot.dof_names)))
            world.physics_sim_view.update_articulations_kinematic()
            physx.update_transformations(False,True,False)
            expected=fk(ROOT/model['urdf'],q)
            for name,path in scene_paths.items():
                prim=stage.GetPrimAtPath(path)
                if not prim:raise ValueError('Missing retained link '+name)
                actual=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                error=float(np.max(abs(actual-expected[name])))
                report['maximum_fk_error_m']=max(report['maximum_fk_error_m'],error)
                if error>3e-5:raise ValueError(f'{name} USD/URDF mismatch {error}')
            if not np.allclose(scan_before,np.asarray(UsdGeom.Xformable(scan_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())),atol=1e-10):
                raise ValueError('Registered drawer moved with the wrist')
            for _ in range(2):app.update()
            report['frames']+=1
        for _ in range(args.frames):app.update()
        if not args.headless:
            from isaacsim.core.rendering_manager import ViewportManager
            ViewportManager.wait_for_viewport(max_frames=120)
            ViewportManager.set_camera_view(ViewportManager.get_camera(),eye=[1.3,-1.2,1.15],target=[.25,-.05,.3])
        if args.screenshot:
            import omni.replicator.core as rep
            from PIL import Image
            stage.GetPrimAtPath('/World/Overview').GetAttribute('focalLength').Set(24.)
            product=rep.create.render_product('/World/Overview',(1440,1080))
            rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product])
            for _ in range(35):app.update()
            rep.orchestrator.step(delta_time=0.0,rt_subframes=4,pause_timeline=True)
            args.screenshot.parent.mkdir(parents=True,exist_ok=True)
            pixels=rgb.get_data()
            if not isinstance(pixels,np.ndarray) or pixels.size==0:raise ValueError('Screenshot capture returned no pixels')
            Image.fromarray(pixels[...,:3]).save(args.screenshot)
        if not args.headless:
            import omni.ui as ui
            panel=ui.Window('Specimen Workcell',width=420,height=220)
            with panel.frame:
                with ui.VStack(spacing=8):
                    ui.Label(args.drawer+' / '+args.robot)
                    ui.Label('Offline replay. No robot or camera connections.',word_wrap=True)
                    ui.Label('Use --replay to show the recorded survey route.\nChoose other drawer/models with launch arguments.',word_wrap=True)
            while app.is_running():app.update();time.sleep(.01)
        report['status']='passed'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
        app.close()


if __name__=='__main__':main()
