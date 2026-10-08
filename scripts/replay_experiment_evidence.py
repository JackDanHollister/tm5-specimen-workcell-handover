"""Verify saved experiment files and replay route/arrival evidence; no device imports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def route_summary(programme):
    """Validate stored visit order and measure planned motion, not hardware throughput."""
    visits = []
    position = 'start'
    seconds = 0.0
    minimum = float('inf')
    labels = set()
    last_target = None
    for batch in programme['batches']:
        for station in batch['stations']:
            if station['label'] in labels:
                raise ValueError('Repeated photographic station')
            labels.add(station['label'])
            for key in station['transitions']:
                edge = programme['transitions'][key]
                if edge['from'] != position:
                    raise ValueError('Discontinuous route graph')
                move = edge['move']
                start = np.asarray(move['start_q_rad'], float)
                target = np.asarray(move['target_q_rad'], float)
                if start.shape != (6,) or target.shape != (6,) or not np.isfinite(np.r_[start, target]).all():
                    raise ValueError('Invalid six-axis route')
                # Saved nodes retain more digits than the rounded wire target.
                if last_target is not None and np.max(np.abs(start-last_target)) > np.radians(.000011):
                    raise ValueError('Discontinuous joint endpoints')
                expected = move['nominal_duration_s']/(move['project_speed_percent']/100)
                if not np.isclose(expected, move['effective_duration_s'], atol=1e-9, rtol=0):
                    raise ValueError('Override/duration mismatch')
                if move['hardware_execution_authorised'] or move['physical_validation_complete']:
                    raise ValueError('Offline route unexpectedly claims physical qualification')
                if hashlib.sha256(move['command'].encode()).hexdigest() != move['command_sha256']:
                    raise ValueError('Changed controller command')
                position = edge['to']
                last_target = target
                seconds += expected
                minimum = min(minimum, edge['recorded_minimum_scene_clearance_mm'])
                visits.append(key)
            if position != station['label']:
                raise ValueError('Capture is not at its requested station')
    if visits != programme['executed_transition_sequence']:
        raise ValueError('Changed ordered transition list')
    if not np.isclose(seconds, programme['estimated_motion_seconds'], atol=1e-9, rtol=0):
        raise ValueError('Changed timing total')
    return {'views': len(labels), 'batches': len(programme['batches']), 'controller_moves': len(visits),
            'planned_motion_seconds': seconds, 'recorded_minimum_clearance_mm': minimum,
            'hardware_time_measured': False, 'collision_requalified': False}


def arrival_summary(record):
    """Recalculate endpoint evidence without claiming queue completion or collision clearance."""
    target = np.asarray(record['segments'][-1]['target'], float)
    target_rotation = Rotation.from_euler('xyz', target[3:], degrees=True)
    errors, consecutive = [], 0
    first_three = None
    began = record['feedback'][0]['sample']['received_monotonic_s']
    for row in record['feedback']:
        sample = row['sample']
        pose = np.asarray(sample['feedback']['tool0_pose'], float)
        translation = float(np.linalg.norm(pose[:3]*1000-target[:3]))
        rotation = float(np.degrees((target_rotation.inv()*Rotation.from_euler('xyz', pose[3:])).magnitude()))
        valid = translation <= .1 and rotation <= .05
        consecutive = consecutive+1 if valid else 0
        if consecutive >= 3 and first_three is None:
            first_three = sample['received_monotonic_s']-began
        errors.append((translation, rotation, valid))
    return {'original_status': record['status'], 'original_error': record['error'],
            'feedback_samples': len(errors), 'endpoint_samples_within_original_tolerance': sum(x[2] for x in errors),
            'first_three_settled_samples_after_s': first_three,
            'last_translation_error_mm': errors[-1][0], 'last_rotation_error_deg': errors[-1][1],
            'last_old_segment_index': record['feedback'][-1]['route_segment'],
            'required_final_segment_index': len(record['segments'])-1,
            'limits': 'Pose-only replay; no queue completion, geometry, synchronization or physical qualification inferred'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/experiment-replay.json')
    args = parser.parse_args()
    index = json.loads((ROOT/'configs/experiments.json').read_text())
    for row in index['files']:
        path = (ROOT/row['path']).resolve()
        if ROOT not in path.parents or not path.is_file() or path.stat().st_size != row['bytes']:
            raise ValueError('Missing/invalid evidence: '+row['path'])
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                raise ValueError('Changed evidence: '+row['path'])
    records = ROOT/'data/experiments/records'
    programme = json.loads((ROOT/'examples/prior-rapid-programme.json').read_text())
    failed = json.loads((records/'20261005-close-pivots-live-v1/batches/close_01_chunk_02.json').read_text())
    completed = json.loads((records/'20261006-close-pivots-live-v1/run_summary.json').read_text())
    cold = json.loads((records/'rapid-batch-benchmarks/20261006-v3/benchmark.json').read_text())
    historical = json.loads((records/'rapid-batch-benchmarks/historical-v2/benchmark.json').read_text())
    result = {'verified_original_files': len(index['files']), 'recorded_sessions': len(index['sessions']),
              'rapid_route': route_summary(programme), 'false_stall_endpoint': arrival_summary(failed),
              'later_complete_capture': {k: completed[k] for k in ('view_count','batch_count','motion_capture_seconds','fit_status','timing')},
              'current_saved_image_benchmark': cold, 'historical_saved_image_benchmark': historical,
              'hardware_contact': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('current_saved_image_benchmark','historical_saved_image_benchmark','later_complete_capture')}, indent=2))


if __name__ == '__main__':
    main()
