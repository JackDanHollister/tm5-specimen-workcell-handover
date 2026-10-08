"""Offline invariants for shaft sign, rigid rotation and TCP preservation."""
import unittest
import numpy as np
from frames import transform
from orientation_trial import orientation_trial


class OrientationTrialTests(unittest.TestCase):
    def setUp(self):
        self.flange = transform([.415, -.033, .385], [np.pi, 0, np.pi / 2])
        self.offset = np.array([0, 0, .17085])

    def test_axis_alignment_preserves_lifted_pinch_and_proper_rotation(self):
        axis = np.array([.43, .036, .902]); axis /= np.linalg.norm(axis)
        result = orientation_trial(axis, self.flange, self.offset)
        target = np.array(result['aligned_flange_base_m'])
        np.testing.assert_allclose(target[:3, 2], -axis, atol=1e-12)
        np.testing.assert_allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(target[:3, :3]), 1)
        initial_pinch = self.flange[:3, 3] + self.flange[:3, :3] @ self.offset
        np.testing.assert_allclose(target[:3, 3] + target[:3, :3] @ self.offset,
                                   initial_pinch + [0, 0, .03], atol=1e-12)
        self.assertFalse(result['motion_authority'])

    def test_undirected_line_sign_does_not_reverse_gripper(self):
        axis = np.array([.2, .1, 1])
        a = orientation_trial(axis, self.flange, self.offset)
        b = orientation_trial(-axis, self.flange, self.offset)
        np.testing.assert_allclose(a['aligned_flange_base_m'], b['aligned_flange_base_m'])

    def test_already_aligned_keeps_wrist_roll(self):
        result = orientation_trial([0, 0, 1], self.flange, self.offset)
        np.testing.assert_allclose(np.array(result['aligned_flange_base_m'])[:3, :3],
                                   self.flange[:3, :3], atol=1e-12)

    def test_invalid_and_ambiguous_geometry_rejected(self):
        for axis in ([0, 0, 0], [np.nan, 0, 1], [1, 0, 0]):
            with self.assertRaises(ValueError):
                orientation_trial(axis, self.flange, self.offset)
        with self.assertRaises(ValueError):
            orientation_trial([0, 0, 1], np.eye(4), self.offset)
        with self.assertRaises(ValueError):
            orientation_trial([0, 0, 1], self.flange, self.offset, lift_m=-.03)


if __name__ == '__main__':
    unittest.main()
