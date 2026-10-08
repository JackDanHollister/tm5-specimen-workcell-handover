"""Offline contract tests for the archived route and arrival interpretation."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('experiment_replay', ROOT/'scripts/replay_experiment_evidence.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ExperimentEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.route = json.loads((ROOT/'examples/prior-rapid-programme.json').read_text())

    def test_known_programme_is_connected_and_override_adjusted(self):
        result = replay.route_summary(self.route)
        self.assertEqual((result['views'], result['batches'], result['controller_moves']), (11, 3, 18))
        self.assertAlmostEqual(result['planned_motion_seconds'], 35.75342222222222)
        self.assertFalse(result['hardware_time_measured'])

    def test_broken_order_cannot_look_like_a_valid_route(self):
        self.route['batches'][0]['stations'][0]['transitions'].reverse()
        with self.assertRaisesRegex(ValueError, 'Discontinuous route graph'):
            replay.route_summary(self.route)

    def test_override_blind_duration_is_rejected(self):
        first = self.route['executed_transition_sequence'][0]
        move = self.route['transitions'][first]['move']
        move['effective_duration_s'] = move['nominal_duration_s']
        with self.assertRaisesRegex(ValueError, 'Override/duration mismatch'):
            replay.route_summary(self.route)

    def test_changed_wire_command_does_not_pass_provenance(self):
        first = self.route['executed_transition_sequence'][0]
        self.route['transitions'][first]['move']['command'] += '\r\nQueueTag(2)'
        with self.assertRaisesRegex(ValueError, 'Changed controller command'):
            replay.route_summary(self.route)


if __name__ == '__main__':
    unittest.main()
