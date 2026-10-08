"""URDF frame transforms in metres/radians for the packaged robot variants."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def rigid(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    value[:3, 3] = xyz
    return value


def fk(urdf, joints):
    q = np.asarray(joints, float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('Six finite joint positions in radians required')
    frames = {'base': np.eye(4)}
    pending = list(ET.parse(Path(urdf)).getroot().findall('joint'))
    while pending:
        ready = [j for j in pending if j.find('parent').get('link') in frames]
        if not ready:
            raise ValueError('Disconnected or cyclic URDF')
        for joint in ready:
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            t = rigid(xyz, rpy)
            name = joint.get('name')
            if name in [f'joint_{i}' for i in range(1, 7)]:
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                t[:3, :3] = t[:3, :3] @ Rotation.from_rotvec(axis * q[int(name[-1])-1]).as_matrix()
            frames[joint.find('child').get('link')] = frames[joint.find('parent').get('link')] @ t
            pending.remove(joint)
    return frames


def limits(urdf):
    tree = ET.parse(urdf).getroot()
    return {f'joint_{i}': dict(tree.find(f"joint[@name='joint_{i}']/limit").attrib) for i in range(1, 7)}

