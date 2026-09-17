"""Headless physics regression: both fingers, grasp retention and release.

Only rendering and HTTP transport are replaced; the EEF solver, contacts,
actuators, equalities, and every physical integration step remain real.
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent / 'harness'))
import simsvc
import test_pick_place as scenario


class GripperStability(unittest.TestCase):
    def test_full_pick_place_and_finger_sync(self):
        samples = []
        physical = []
        original_step = simsvc.mujoco.mj_step
        def step(model, data):
            original_step(model, data)
            q = [float(data.qpos[model.joint(f"left_joint{i}").qposadr[0]]) for i in (7,8)]
            physical.append(q)
        phase = ['initial']
        session = []
        def capture(s):
            samples.append({'phase': phase[0], 'q': s.finger_state()['left']['q'],
                            'grasped': s.grasped['left']})
        def start(url, payload):
            session.append(simsvc.Session(payload['layout'], payload['instruction']))
            return {'session_id': 'headless'}
        def observe(sid):
            s = session[0]
            return {**s._status(), 'ee': s.ee_poses()}, {}
        def execute(sid, step, route, payload, reason):
            s = session[0]; phase[0] = reason
            return s._status() if route == 'finish' else s.eef(payload['goals'], payload['steps'])
        with patch.object(simsvc.mujoco, 'mj_step', step), \
             patch.object(simsvc.mujoco, 'Renderer', MagicMock()), \
             patch.object(simsvc.Session, 'capture', capture), \
             patch.object(scenario.client, 'http', start), \
             patch.object(scenario.client, 'observe', observe), \
             patch.object(scenario.client, 'execute', execute):
            scenario.main()  # asserts real contact acquisition and bottle in bin
            q = np.asarray([r['q'] for r in samples])
            sync_mm = np.max(np.abs(q[:, 0]-q[:, 1]))*1000
            self.assertLess(sync_mm, 0.3, 'two fingers lost synchronization')
            metrics = {'max_sync_error_mm': float(sync_mm)}
            for name in ('lift', 'carry'):
                rows = [r for r in samples if r['phase'] == name]
                self.assertTrue(all(r['grasped'] == 'bottle0' for r in rows))
                a = np.asarray([r['q'] for r in rows])
                jump = float(np.max(np.abs(np.diff(a, axis=0)))*1000)
                span = float(np.max(np.ptp(a, axis=0))*1000)
                self.assertLess(jump, 0.5, name+' finger jump >0.5mm/frame')
                self.assertLess(span, 1.0, name+' opening drift >1mm')
                metrics[name] = {'max_frame_delta_mm': jump, 'opening_span_mm': span}
            self.assertIsNone(session[0].grasped['left'])
            # With fixed arm/gripper commands, check both hands after settling.
            s = session[0]; row = s.state14(); row[6] = row[13] = 1.0
            for _ in range(50): s._step(row)
            held = []
            for _ in range(50):
                s._step(row)
                held.append([v for side in s.finger_state().values() for v in side['q']])
            hold_mm = float(np.max(np.ptp(held, axis=0))*1000)
            self.assertLess(hold_mm, 0.1, 'stationary open jaws oscillate')
            metrics['stationary_span_mm'] = hold_mm
            substep_sync = float(np.max(np.abs(np.diff(np.asarray(physical), axis=1)))*1000)
            self.assertLess(substep_sync, 0.3, 'finger sync failed between video frames')
            metrics['physics_substep_sync_mm'] = substep_sync
            print(json.dumps(metrics, indent=2))


if __name__ == '__main__': unittest.main(verbosity=2)
