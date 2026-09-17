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
    def test_left_pick_place_and_finger_sync(self):
        self.check_side("left")

    def test_right_pick_place_and_finger_sync(self):
        self.check_side("right")

    def check_side(self, side):
        self.assertFalse(simsvc.GRASP_ASSIST, "physical acceptance requires assist off")
        samples = []
        physical = []
        original_step = simsvc.mujoco.mj_step
        def step(model, data):
            original_step(model, data)
            q = [float(data.qpos[model.joint(f"{side}_joint{i}").qposadr[0]]) for i in (7,8)]
            physical.append(q)
        phase = ['initial']
        session = []
        def capture(s):
            self.assertFalse(any(s.data.eq_active[i] for i in range(s.model.neq) if s.model.eq_type[i] == simsvc.mujoco.mjtEq.mjEQ_WELD))
            samples.append({'phase': phase[0], 'q': s.finger_state()[side]['q'],
                            'grasped': s.grasped[side],
                            'bottle': s.data.xpos[s.obj_body_ids['bottle0']].copy(),
                            'penetration_mm': max([max(0.0,-c.dist)*1000 for c in s.data.contact if s.obj_body_ids['bottle0'] in (s.model.geom_bodyid[c.geom1],s.model.geom_bodyid[c.geom2])],default=0.0)})
        def start(url, payload):
            session.append(simsvc.Session(payload['layout'], payload['instruction']))
            model = session[0].model
            for hand in ('left','right'):
                for i in range(1,9):
                    body = model.body(f'{hand}_link{i}')
                    for g in range(int(body.geomadr[0]),int(body.geomadr[0]+body.geomnum[0])):
                        self.assertNotEqual(int(model.geom_contype[g]),0,'robot collision disabled')
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
            scenario.main(side)  # asserts real contact acquisition and bottle in bin
            q = np.asarray([r['q'] for r in samples])
            sync_mm = np.max(np.abs(q[:, 0]-q[:, 1]))*1000
            self.assertLess(sync_mm, 0.3, 'two fingers lost synchronization')
            metrics = {'side': side, 'max_sync_error_mm': float(sync_mm)}
            for name in ('lift', 'hold 3 seconds', 'carry'):
                rows = [r for r in samples if r['phase'] == name]
                fraction = sum(r['grasped'] == 'bottle0' for r in rows)/len(rows)
                print('bilateral fraction',side,name,fraction)
                self.assertGreaterEqual(fraction, 1.0 if name == "hold 3 seconds" else 0.98, "insufficient bilateral contact")
                a = np.asarray([r['q'] for r in rows])
                jump = float(np.max(np.abs(np.diff(a, axis=0)))*1000)
                span = float(np.max(np.ptp(a, axis=0))*1000)
                self.assertLess(jump, 0.5, name+' finger jump >0.5mm/frame')
                if name == 'hold 3 seconds':
                    self.assertLess(span, 0.5, 'holding jaw drift >0.5mm')
                    heights = [r['bottle'][2] for r in rows]
                    self.assertGreater(min(heights), 1.0, 'bottle was not held airborne')
                    self.assertLess(np.ptp(heights), 0.005, 'bottle slipped >5mm while holding')
                penetration = max(r['penetration_mm'] for r in rows)
                self.assertLess(penetration, 1.0, 'bottle penetration >1mm')
                metrics[name] = {'max_frame_delta_mm': jump, 'opening_span_mm': span, 'max_penetration_mm': penetration, 'bilateral_contact_fraction': fraction}
            self.assertIsNone(session[0].grasped[side])
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
