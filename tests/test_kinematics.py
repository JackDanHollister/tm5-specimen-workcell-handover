from pathlib import Path
import tempfile
import unittest
import numpy as np
from workcell_kit.kinematics import fk,rigid


class KinematicsTests(unittest.TestCase):
    def test_rotated_fixed_camera_offset_follows_flange(self):
        joints=''.join(f'<joint name="joint_{i}" type="revolute"><parent link="l{i-1}"/><child link="l{i}"/><axis xyz="0 0 1"/><origin xyz="0 0 0" rpy="0 0 0"/></joint>' for i in range(1,7))
        text='<robot><joint name="base_fixed" type="fixed"><parent link="base"/><child link="l0"/></joint>'+joints+'<joint name="camera" type="fixed"><parent link="l6"/><child link="camera"/><origin xyz="0.1 0 0" rpy="0 0 0"/></joint></robot>'
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'robot.urdf';path.write_text(text)
            a=fk(path,np.zeros(6))['camera'];b=fk(path,[np.pi/2,0,0,0,0,0])['camera']
            np.testing.assert_allclose(a[:3,3],[.1,0,0],atol=1e-12)
            np.testing.assert_allclose(b[:3,3],[0,.1,0],atol=1e-12)

    def test_invalid_pose_and_disconnected_tree_are_rejected(self):
        with self.assertRaises(ValueError):fk('unused',[0,0])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'robot.urdf';path.write_text('<robot><joint name="x"><parent link="absent"/><child link="b"/></joint></robot>')
            with self.assertRaises(ValueError):fk(path,np.zeros(6))

    def test_rigid_convention(self):
        t=rigid([1,2,3],[0,0,np.pi/2])
        np.testing.assert_allclose(t@np.array([1,0,0,1]),[1,3,3,1],atol=1e-12)
