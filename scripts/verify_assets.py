"""Verify the complete downloaded kit using package-relative file identities."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--usd',action='store_true');parser.add_argument('--without-models',action='store_true');args=parser.parse_args()
    rows=json.loads((ROOT/'configs/payload_files.json').read_text())
    checked=0
    for row in rows:
        if args.without_models and row['path'].startswith('models/'):continue
        path=(ROOT/row['path']).resolve()
        if ROOT.resolve() not in path.parents:raise ValueError('Payload escapes package')
        if not path.is_file() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Payload missing or changed: '+row['path'])
        checked+=1
    robots=json.loads((ROOT/'configs/robots.json').read_text())
    drawers=json.loads((ROOT/'configs/drawers.json').read_text())
    assert len(robots)==4 and len(drawers)==6
    for robot in robots:
        tree=ET.parse(ROOT/robot['urdf']).getroot();links={x.get('name') for x in tree.findall('link')}
        assert ('photoneo_camera' in links)==robot['scanner_present']
        assert ('onrobot_2fg7_origin' in links)==robot['gripper_present']
        assert 'onrobot_qc_robot_side_link' in links and 'eih_camera_optical' in links
        for mesh in tree.findall('.//mesh'):
            path=((ROOT/robot['urdf']).parent/mesh.get('filename')).resolve()
            assert path.is_file() and ROOT.resolve() in path.parents
    images=0;scans=0;points=0
    for drawer in drawers:
        folder=ROOT/drawer['folder']
        assert (folder/'eih/image.png').is_file();images+=1
        for name in ('centre','y_minus100','y_plus100'):
            meta=json.loads((folder/'scans'/name/'scan_001.json').read_text())
            assert meta['pose_association_accepted'];scans+=1;points+=meta['valid_points']
    assert (images,scans,points)==(6,18,9839520)
    if args.usd:
        from pxr import Usd,UsdUtils
        for scene in json.loads((ROOT/'configs/scenes.json').read_text()):
            stage=Usd.Stage.Open(str(ROOT/scene['usd']));assert stage
            layers,assets,missing=UsdUtils.ComputeAllDependencies(str(ROOT/scene['usd']))
            if missing:raise ValueError('Unresolved USD assets: '+str(missing))
            for layer in layers:
                if layer.realPath and ROOT.resolve() not in Path(layer.realPath).resolve().parents:
                    raise ValueError('USD depends on files outside the package')
    print(f'Verified {checked} payload files; 4 robot models, 6 drawers, 6 photos, 18 scans, {points} overlapping observations')


if __name__=='__main__':main()
