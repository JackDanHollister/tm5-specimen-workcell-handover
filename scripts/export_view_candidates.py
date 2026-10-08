"""Translate historical specimen-relative viewing patterns to a new framing point."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target-m',nargs=3,type=float,required=True)
    p.add_argument('--reference-specimen',default='4')
    p.add_argument('--robot',default='scanner_no_gripper')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/view-candidates.json')
    args=p.parse_args()
    patterns=json.loads((ROOT/'data/pins/view_patterns.json').read_text())
    models=json.loads((ROOT/'configs/robots.json').read_text())
    model=next(x for x in models if x['name']==args.robot)
    handeye=np.asarray(model['T_flange_eih_m'])
    rows=[]
    for row in patterns:
        if row['specimen_id']!=args.reference_specimen:continue
        camera=np.asarray(row['T_target_eih_m']).copy();camera[:3,3]+=args.target_m
        rows.append({'reference_image':row['id'],'T_base_eih_m':camera.tolist(),
                     'T_base_flange_m':(camera@np.linalg.inv(handeye)).tolist()})
    if not rows:raise ValueError('No supporting views for this reference placement')
    result={'robot':args.robot,'target_base_m':args.target_m,'candidate_views':rows,
            'ik_solved':False,'collision_checked':False,'physical_execution_allowed':False,
            'target_frame_axes':'parallel to robot-base axes; origin at the requested framing point',
            'next_step':'Solve IK, visibility and complete swept-path clearance for the current drawer/model'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Exported {len(rows)} target-relative viewing candidates; these are not executable trajectories')


if __name__=='__main__':main()
