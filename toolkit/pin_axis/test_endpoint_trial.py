"""CPU-only endpoint geometry, censoring and consensus regression tests."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from endpoint_trial import consensus, intervals, project, propose_endpoints, run, sample


def synthetic_profile():
    grid = np.arange(-30., 30.0001, .1)
    pin = ((grid >= 0) & (grid <= 12)).astype(float) * .8
    probabilities = np.zeros((5, len(grid)))
    probabilities[1] = pin
    probabilities[2] = (grid < 0) * .9
    probabilities[0] = 1 - probabilities.sum(axis=0)
    return grid, {"probabilities": probabilities, "contrast": np.where(pin>0,90.,6.),
                  "valid": np.ones(len(grid),bool), "pixels": np.column_stack((grid*10,np.full(len(grid),100))),
                  "broad_occupancy": (grid<1).astype(float)}


class EndpointTests(unittest.TestCase):
    def test_consensus_keeps_outliers_and_reports_empirical_not_accuracy_range(self):
        result=consensus([("a",10), ("b",10.2), ("c",9.8), ("bad",25)])
        self.assertEqual(result["status"],"provisional_consensus")
        self.assertEqual(result["median_mm"],10)
        self.assertEqual(result["other_ids"],["bad"])
        self.assertIn("not_calibrated_accuracy",result["interval_kind"])

    def test_competing_equal_clusters_do_not_become_consensus(self):
        result=consensus([(str(i), v) for i,v in enumerate([0,.1,.2,10,10.1,10.2])])
        self.assertNotEqual(result["status"],"provisional_consensus")

    def test_consensus_members_obey_radius_of_reported_median(self):
        values=[("a",3.4),("b",4.2),("c",5.2),("d",5.3),("e",5.4)]
        result=consensus(values)
        for key,value in values:
            if key in result["support_ids"]:
                self.assertLessEqual(abs(value-result["median_mm"]),result["tolerance_mm"])

    def test_absent_and_nonfinite_candidates_do_not_create_endpoints(self):
        result=consensus([("a",None), ("b",float("nan"))])
        self.assertIsNone(result["median_mm"])
        self.assertEqual(result["support_ids"],[])

    def test_known_head_and_body_boundary(self):
        grid, profile=synthetic_profile()
        result=propose_endpoints(profile,grid,12)
        self.assertAlmostEqual(result["head_candidate_mm"],12,delta=.3)
        self.assertAlmostEqual(result["native_obstruction_mm"],1,delta=.2)
        self.assertAlmostEqual(result["native_span_mm"],11,delta=.5)

    def test_no_visible_pixels_abstains(self):
        grid,profile=synthetic_profile()
        profile["valid"][:]=False
        result=propose_endpoints(profile,grid,12)
        self.assertIsNone(result["head_candidate_mm"])
        self.assertIsNone(result["native_span_mm"])

    def test_search_boundary_is_censored_not_a_short_grasp_length(self):
        grid,profile=synthetic_profile()
        profile["broad_occupancy"][:]=1
        result=propose_endpoints(profile,grid,12)
        self.assertIsNone(result["native_span_mm"])
        self.assertIn("boundary_unknown",result["native_obstruction_reason"])

    def test_native_bright_and_dark_contrast_use_identical_endpoint_contract(self):
        grid,profile=synthetic_profile()
        # profile_view supplies absolute native contrast for either polarity.
        a=propose_endpoints(profile,grid,12)
        profile["contrast"]=np.where(profile["contrast"]>20,120.,7.)
        b=propose_endpoints(profile,grid,12)
        self.assertEqual(a["head_candidate_mm"],b["head_candidate_mm"])

    def test_sampling_pixel_centres_and_outside_image(self):
        field=np.arange(20,dtype=np.float32).reshape(4,5)
        values=sample(field,np.array([[1.,1.],[3.,2.],[-2.,1.]]))
        np.testing.assert_allclose(values[:2],[6,13])
        self.assertTrue(np.isnan(values[2]))

    def test_projection_respects_camera_rotation_not_image_up(self):
        pose=np.eye(4)
        pose[:3,:3]=np.diag([-1.,-1.,1.])
        row={"camera_pose":pose,"K":[[100,0,50],[0,100,50],[0,0,1]],"distortion":[0]*5}
        pixels,depth=project(np.array([[.1,.2,1.],[.1,.3,1.]]),row)
        np.testing.assert_allclose(pixels,[[40,30],[40,20]])
        np.testing.assert_allclose(depth,[1,1])

    def test_short_isolated_regions_are_not_a_long_visible_shaft(self):
        self.assertEqual(intervals(np.array([0,1,0,1,0],bool),np.arange(5.),.5),[])

    def test_cannot_write_into_original_run(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)
            for target in (source,source/"new"):
                with self.assertRaisesRegex(ValueError,"separate"):
                    run(source,target)


if __name__=="__main__":
    unittest.main()
