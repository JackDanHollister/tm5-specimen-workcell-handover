"""Recompute the seven historical shaft fits from saved image observations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'toolkit/pin_axis'))
from multiview import observation, solve


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/pin-replay.json')
    args=parser.parse_args()
    rows=json.loads((ROOT/'data/pins/observations.json').read_text())
    expected=json.loads((ROOT/'data/pins/reconstruction.json').read_text())['specimens']
    results=[]
    for old in expected:
        selected=[r for r in rows if r['specimen_id']==old['specimen_id'] and r['capture_accepted'] and r['geometry']['angle_deg'] is not None]
        observations=[]
        for r in selected:
            if hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()!=r['source_sha256']:
                raise ValueError('Historical image changed: '+r['id'])
            observations.append(observation(r['id'],r['geometry']['endpoints_xy'],np.asarray(r['camera_pose']),np.asarray(r['K']),np.asarray(r['distortion'])))
        fit=solve(observations)
        if fit['axis_base'] is None:raise ValueError('Stored successful fit no longer reproduced')
        a,b=np.asarray(fit['axis_base']),np.asarray(old['primary']['axis_base'])
        delta=float(np.degrees(np.arccos(np.clip(abs(a@b),-1,1))))
        if delta>.001:raise ValueError('Replayed direction differs from original result')
        results.append({'specimen_id':old['specimen_id'],'fit':fit,'original_axis_difference_deg':delta})
    value={'status':'reproduced_saved_observations','placements':len(results),'results':results,
           'physical_pick_allowed':False,'pin_identity_physically_verified':False,'robot_connected':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(f'Reproduced {len(results)} provisional historical pin axes; no inference or hardware')


if __name__=='__main__':main()
