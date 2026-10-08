"""Run cached full-image pin reconstruction in a fresh output directory."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();output=args.output.resolve()
    if output.exists():raise ValueError('Use a fresh derived output directory')
    output.mkdir(parents=True)
    records=json.loads((ROOT/'data/pins/observations.json').read_text())
    manifest=[]
    for record in records:
        manifest.append({'id':record['id'],'specimen_id':record['specimen_id'],
            'path':str(ROOT/record['path']),'coordinates_path':str(ROOT/record['coordinates_path']),
            'role':'multiview_development'})
        shutil.copytree(ROOT/'data/pins/cache'/record['id'],output/'cache'/record['id'])
        # Only the fresh derived cache is rewritten; archived inputs stay unchanged.
        meta_path=output/'cache'/record['id']/'input.json'
        meta=json.loads(meta_path.read_text())
        meta['path']=str(ROOT/record['path']);meta['coordinates_path']=str(ROOT/record['coordinates_path'])
        meta_path.write_text(json.dumps(meta,indent=2)+'\n')
    shutil.copy2(ROOT/'data/pins/cache/model.json',output/'cache/model.json')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'toolkit/pin_axis/reconstruct.py'),
        '--run',str(output),'--calibration',str(ROOT/'data/pins/historical_eih_metadata.json'),
        '--urdf',str(ROOT/'assets/robots/no_scanner_gripper/robot.urdf'),'--fixed-placements-confirmed'],check=True)
    print('Fresh cached reconstruction complete; package evidence untouched')


if __name__=='__main__':main()
