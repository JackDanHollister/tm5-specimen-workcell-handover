"""Known synthetic geometry, frame conventions and calibration selection."""
import unittest

import cv2
import numpy as np

from frames import intrinsics_for, transform
from multiview import linear_solve, observation, solve


def make_views(outlier=False, same_center=False):
    K = np.array([[1400., 0, 640], [0, 1400, 480], [0, 0, 1]])
    anchor = np.array([0.45, 0.04, 0.08])
    axis = np.array([0.2, -0.1, 1.]); axis /= np.linalg.norm(axis)
    views = []
    for i, phi in enumerate(np.linspace(0, 2 * np.pi, 6, endpoint=False)):
        center = anchor + np.array([0.20 * np.cos(phi), 0.20 * np.sin(phi), 0.18])
        if same_center:
            center = anchor + np.array([0.2, 0, 0.18])
        forward = anchor - center; forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1]); right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        T = np.eye(4); T[:3, :3] = np.column_stack((right, down, forward)); T[:3, 3] = center
        # Every camera observes a DIFFERENT segment of the same infinite shaft.
        points = anchor + np.array([-0.035 + i * .002, 0.035 - i * .003])[:, None] * axis
        camera_points = (points - center) @ T[:3, :3]
        pixels = camera_points @ K.T
        pixels = pixels[:, :2] / pixels[:, 2:3]
        if outlier and i == 5:
            pixels += [80, -50]
        views.append(observation(str(i), pixels, T, K, np.zeros(5)))
    return anchor, axis, views


class MultiviewTests(unittest.TestCase):
    def test_different_visible_endpoints_recover_same_3d_line(self):
        anchor, axis, views = make_views()
        result = solve(views)
        self.assertEqual(result["status"], "provisional_consensus")
        self.assertEqual(len(result["inlier_ids"]), 6)
        recovered = np.array(result["axis_base"])
        point = np.array(result["point_on_axis_base_m"])
        self.assertGreater(abs(recovered @ axis), 0.999999)
        self.assertLess(np.linalg.norm(np.cross(anchor - point, recovered)), 1e-7)
        self.assertLess(max(result["leave_one_view_out_residual_px"]), 1e-5)

    def test_bad_detection_is_excluded(self):
        anchor, axis, views = make_views(outlier=True)
        result = solve(views)
        self.assertEqual(result["status"], "provisional_consensus")
        self.assertEqual(len(result["inlier_ids"]), 5)
        self.assertNotIn("5", result["inlier_ids"])
        self.assertGreater(abs(np.array(result["axis_base"]) @ axis), .99999)

    def test_same_camera_center_is_not_a_multiview_baseline(self):
        _, _, views = make_views(same_center=True)
        self.assertEqual(solve(views)["status"], "insufficient_consistent_views")
        with self.assertRaisesRegex(ValueError, "parallel"):
            linear_solve(views[:2])

    def test_intrinsics_require_exact_focus_and_resolution(self):
        camera = {"factory_intrinsics": [{"focus": 5, "width_px": 2592, "height_px": 1944,
                                          "camera_matrix": np.eye(3).tolist(), "distortion": [0]*5}]}
        intrinsics_for(camera, 5, (2592, 1944))
        with self.assertRaises(ValueError):
            intrinsics_for(camera, 4, (2592, 1944))
        with self.assertRaises(ValueError):
            intrinsics_for(camera, 5, (1280, 960))

    def test_rpy_is_fixed_axis_xyz_with_camera_offset_rotated(self):
        T = transform([0.4, 0, 0.2], [0, 0, np.pi / 2])
        camera = T @ transform([0.075, 0, 0.04], [0, 0, 0])
        np.testing.assert_allclose(camera[:3, 3], [0.4, 0.075, 0.24], atol=1e-12)

    def test_reprojection_curve_stays_inside_calibrated_field(self):
        from multiview_report import axis_curve
        from multiview import projected_line
        anchor, axis, views = make_views()
        obs = views[0]
        distortion = np.array([.15, -.28, .0002, -.0001, -.106])
        curve = axis_curve(anchor, axis, obs["pose"], obs["K"], distortion, (1280, 960))
        self.assertGreater(len(curve), 20)
        self.assertLess(abs(curve).max(), 1500)
        undistorted = cv2.undistortPoints(curve[:, None], obs["K"], distortion, P=obs["K"])[:, 0]
        line = projected_line(anchor, axis, obs)
        error = np.column_stack((undistorted, np.ones(len(undistorted)))) @ line
        self.assertLess(abs(error).max(), .03)


if __name__ == "__main__":
    unittest.main()
