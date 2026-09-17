# coding: utf-8
"""Current conversation-policy interface. Never chooses a model or reads old runs."""
import argparse
import fcntl
import base64
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import uuid
import numpy as np
from PIL import Image

MAX_DECISIONS = 180
ROOT = Path(__file__).resolve().parent
SAFE = {'t', 'done', 'success', 'remaining_steps', 'termination', 'state', 'ee',
        'instruction', 'observation_id', 'error'}
CAMERAS = {'cam_base': 'cam_high', 'left_cam_wrist': 'cam_left_wrist',
           'right_cam_wrist': 'cam_right_wrist'}


def public_state(value):
    return {k: v for k, v in value.items() if k in SAFE}


class Controller:
    def __init__(self, session=None, root=ROOT, sim=None, pi=None):
        if session is not None and not re.fullmatch(r'[0-9a-f]{12}', session):
            raise ValueError('Invalid session ID')
        self.root, self.session = Path(root), session
        self.sim = (sim or os.environ.get('ASTRA_SIM', 'http://127.0.0.1:8763')).rstrip('/')
        self.pi = (pi or os.environ.get('ASTRA_PI05', 'http://127.0.0.1:8642')).rstrip('/')
        self.pending = []
        self.directory = self.root / 'runs' / session / 'policy' if session else None
        if self.directory:
            if not self.directory.is_dir():
                raise ValueError('Session was not started by eval_control.py; start a new verified episode')
            config = json.loads((self.directory / 'manifest.json').read_text())
            if config['sim'] != self.sim or config['pi'] != self.pi:
                raise ValueError('Endpoint differs from this episode manifest')
            self.mode = config['mode']

    def log(self, event):
        event = {'wall_time': time.time(), **event}
        if self.directory is None:
            self.pending.append(event)
        else:
            with (self.directory / 'trace.jsonl').open('a') as f:
                f.write(json.dumps(event, allow_nan=False) + '\n')

    def http(self, base, route, data=None):
        request = urllib.request.Request(base + route,
            data=None if data is None else json.dumps(data, allow_nan=False).encode(),
            headers={'Content-Type': 'application/json'})
        start, tick, code = time.time(), time.perf_counter_ns(), None
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                code = response.status
                result = json.load(response)
        except urllib.error.HTTPError as error:
            code = error.code
            result = json.loads(error.read())
        finally:
            self.log({'type': 'http_timing', 'route': route, 'start_wall_time': start,
                      'duration_ms': (time.perf_counter_ns() - tick) / 1e6,
                      'http_status': code, 'request_id': (data or {}).get('request_id')})
        # Do not propagate raw service responses containing privileged positions.
        if code != 200:
            self.log({'type': 'http_error', 'response': public_state(result)})
            raise RuntimeError(f'HTTP {code}: {json.dumps(public_state(result))}')
        return result

    def session_http(self, route, data=None):
        return self.http(self.sim, f'/session/{self.session}/{route}', data)

    def start(self, task, layout, mode):
        if self.session is not None:
            raise ValueError('start cannot reuse an episode')
        health = self.http(self.sim, '/health')
        if not health.get('ok') or not str(health.get('version', '')).startswith('astra-isolated-v'):
            raise ValueError('Not the current isolated simulation service')
        if mode in ('hybrid', 'pi05'):
            if not self.http(self.pi, '/health').get('ok'):
                raise ValueError('Pi service not ready')
        name = 'put_bottles_into_dustbin' if task == 'put_bottles' else task
        instruction = ('Pick up the bottles and throw them into the dustbin, using handover when needed.'
                       if task == 'put_bottles' else 'Sort the objects by category into the three baskets.')
        definition = json.loads((self.root / 'layouts' / f'{name}_{layout}.json').read_text())
        out = self.http(self.sim, '/session', {'layout': definition, 'task': task, 'instruction': instruction})
        sid = out['session_id']
        if not re.fullmatch(r'[0-9a-f]{12}', sid):
            raise ValueError('Invalid server session ID')
        # Same-Mac simulator: verify its reset evidence before allowing control.
        reset = json.loads((Path(out['record_dir']) / 'reset_check.json').read_text())
        required = {'step_zero','joints_home','velocities_zero','controls_match_joints',
                    'objects_at_layout','no_grasp','no_weld','clean_episode'}
        if out['t'] != 0 or not reset.get('passed') or not all(reset.get('checks', {}).get(k) is True for k in required):
            raise ValueError('Reset verification failed; no actions submitted')
        self.session, self.mode = sid, mode
        self.directory = self.root / 'runs' / sid / 'policy'
        self.directory.mkdir(parents=True, exist_ok=False)
        manifest = {'session_id': sid, 'mode': mode, 'task': task, 'layout': layout,
                    'sim': self.sim, 'pi': self.pi, 'reset': reset, 'health': health,
                    'model': 'external controller; not selected by this CLI',
                    'context_isolation': 'Caller must start a fresh GPT task with no inherited history',
                    'observation_allowlist': sorted(SAFE), 'pi_camera_keys': CAMERAS,
                    'max_decisions': MAX_DECISIONS}
        (self.directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        pending, self.pending = self.pending, []
        for event in pending:
            self.log(event)
        return {'session_id': sid, 'mode': mode, 'reset_passed': True, 'policy_dir': str(self.directory)}

    def observe(self):
        begin = time.perf_counter_ns()
        raw = self.session_http('observe')
        tick = time.perf_counter_ns()
        safe = public_state(raw)
        directory = self.directory / 'observations' / f"{raw['t']:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        safe['images'] = {}
        for camera, encoded in raw['images'].items():
            if camera not in CAMERAS:
                continue
            dest = directory / (camera + '.png')
            arr = np.frombuffer(base64.b64decode(encoded), np.uint8).reshape(raw['shapes'][camera])
            Image.fromarray(arr).save(dest)
            safe['images'][camera] = str(dest)
        (directory / 'observation.json').write_text(json.dumps(safe, indent=2))
        self.log({'type': 'observation', 'observation': safe,
                  'decode_save_ms': (time.perf_counter_ns() - tick) / 1e6,
                  'total_ms': (time.perf_counter_ns() - begin) / 1e6})
        return safe

    def infer(self):
        if self.mode == 'direct':
            raise ValueError('direct episode must not call Pi')
        raw = self.session_http('observe')
        if raw['done']:
            raise ValueError('Episode is finished')
        payload = {'images': {v: raw['images'][k] for k, v in CAMERAS.items()},
                   'shapes': {v: raw['shapes'][k] for k, v in CAMERAS.items()},
                   'state': raw['state'], 'prompt': raw['instruction']}
        tick = time.perf_counter_ns()
        result = self.http(self.pi, '/infer', payload)
        elapsed = (time.perf_counter_ns() - tick) / 1e6
        original = np.asarray(result['actions'], dtype=float)
        if original.shape == (1,50,14):
            original = original[0]
        if original.shape != (50,14) or not np.isfinite(original).all():
            raise ValueError('Pi actions must be finite 50x14')
        actions = original.copy()
        actions[:, [6,13]] = np.clip(actions[:, [6,13]], 0, 1)
        data = {'observation_id': raw['observation_id'], 't': raw['t'],
                'actions': actions.tolist(), 'service_ms': result.get('ms'),
                'roundtrip_ms': elapsed, 'gripper_clips': int(np.count_nonzero(actions != original))}
        dest = self.directory / f"proposal_{raw['t']:06d}.json"
        dest.write_text(json.dumps(data, allow_nan=False))
        summary = {k: v for k, v in data.items() if k != 'actions'}
        summary.update(proposal_path=str(dest), first_action=actions[0].tolist(), shape=[50,14])
        self.log({'type': 'pi_inference', 'summary': summary})
        return summary

    def execute(self, route, obs, payload, reason):
        if obs['done'] and route != 'finish':
            raise ValueError('Episode is finished')
        if route != 'finish':
            # Persist across CLI invocations; rejected/uncertain submissions
            # count as attempts. finish always remains available at the cap.
            with (self.directory / 'decision_count').open('a+') as counter:
                fcntl.flock(counter, fcntl.LOCK_EX)
                counter.seek(0)
                count = int(counter.read() or '0')
                if count >= MAX_DECISIONS:
                    return self.execute('finish', obs, {}, 'decision_budget_exhausted:180')
                counter.seek(0); counter.truncate(); counter.write(str(count + 1)); counter.flush()
            self.log({'type': 'decision_budget', 'decision_index': count, 'limit': MAX_DECISIONS})
        body = {'request_id': uuid.uuid4().hex, 'expected_step': obs['t'],
                **payload, 'decision_summary': reason}
        self.log({'type': 'action_request', 'route': route, 'request': body})
        result = public_state(self.session_http(route, body))
        self.log({'type': 'action_response', 'request_id': body['request_id'], 'response': result})
        return result

    def follow(self, steps, reason):
        if self.mode == 'direct':
            raise ValueError('direct episode cannot follow Pi')
        obs = self.session_http('state')
        if type(steps) is not int or not 1 <= steps <= min(15, obs['remaining_steps']):
            raise ValueError('follow steps must be 1..15 and within remaining budget')
        proposal = json.loads((self.directory / f"proposal_{obs['t']:06d}.json").read_text())
        if proposal['observation_id'] != obs['observation_id']:
            raise ValueError('Stale Pi proposal')
        path = self.directory / f"proposal_{obs['t']:06d}.json"
        result = None
        try:
            result = self.execute('act', obs, {'joints': proposal['actions'][:steps]}, reason)
            return result
        finally:
            executed = None if result is None else result['t'] - obs['t']
            self.log({'type': 'chunk_discard', 'observation_id': proposal['observation_id'],
                      'requested_steps': steps, 'executed_steps': executed,
                      'discarded_steps': None if executed is None else len(proposal['actions']) - executed,
                      'planned_suffix_steps': len(proposal['actions']) - steps,
                      'reusable': False})
            path.replace(path.with_suffix('.invalidated.json'))

    def eef(self, payload):
        if self.mode == 'pi05':
            raise ValueError('pi05-only episode cannot use GPT EEF corrections')
        obs = self.session_http('state')
        if payload.get('expected_step') != obs['t']:
            raise ValueError('Stale EEF observation')
        goals = payload.get('goals', {})
        if set(goals) != {'left', 'right'}:
            raise ValueError('Both left and right measured-world targets required')
        steps = payload.get('steps', 1)
        if type(steps) is not int or not 1 <= steps <= min(5, obs['remaining_steps']):
            raise ValueError('EEF tracking steps must be 1..5 within remaining budget')
        for side, goal in goals.items():
            xyz = np.asarray(goal['xyz'], float); quat = np.asarray(goal['quat_wxyz'], float)
            if xyz.shape != (3,) or quat.shape != (4,) or not np.isfinite(xyz).all() or not np.isfinite(quat).all():
                raise ValueError('Invalid finite EEF pose')
            norm = np.linalg.norm(quat)
            if abs(norm - 1) > 1e-3 or not 0 <= float(goal['gripper']) <= 1:
                raise ValueError('Invalid quaternion or gripper')
            current = obs['ee'][side]
            angle = 2 * np.arccos(np.clip(abs(np.dot(quat/norm, current['quat_wxyz'])), 0, 1))
            if np.linalg.norm(xyz - current['xyz']) > .05 or angle > .35:
                raise ValueError('EEF target exceeds 5cm/0.35rad from current measured pose')
        # A correction also discards any unexecuted Pi proposal at this step.
        path = self.directory / f"proposal_{obs['t']:06d}.json"
        try:
            return self.execute('eef', obs, {'goals': goals, 'steps': steps}, payload['decision_summary'])
        finally:
            if path.exists():
                proposal = json.loads(path.read_text())
                self.log({'type': 'chunk_discard', 'observation_id': proposal['observation_id'],
                          'executed_steps': 0, 'discarded_steps': len(proposal['actions']),
                          'reason': 'eef_correction', 'reusable': False})
                path.replace(path.with_suffix('.invalidated.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start','observe','infer','follow','eef','finish'])
    parser.add_argument('--session')
    parser.add_argument('--mode', choices=['direct','hybrid','pi05'], default='direct')
    parser.add_argument('--task', choices=['put_bottles','classify_objects'], default='put_bottles')
    parser.add_argument('--layout', type=int, choices=range(5), default=0)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--payload', type=Path)
    parser.add_argument('--reason')
    args = parser.parse_args()
    if args.command != 'start' and not args.session:
        parser.error('--session required; create a fresh episode with start')
    if args.command in ('follow','finish') and not args.reason:
        parser.error('--reason required')
    if args.command == 'eef' and args.payload is None:
        parser.error('--payload required')
    controller = Controller(args.session)
    if args.command == 'start':
        out = controller.start(args.task, args.layout, args.mode)
    elif args.command == 'eef':
        out = controller.eef(json.loads(args.payload.read_text()))
    elif args.command == 'follow':
        out = controller.follow(args.steps, args.reason)
    elif args.command == 'finish':
        out = controller.execute('finish', controller.session_http('state'), {}, args.reason)
    else:
        out = getattr(controller, args.command)()
    print(json.dumps(out, allow_nan=False))


if __name__ == '__main__':
    main()
