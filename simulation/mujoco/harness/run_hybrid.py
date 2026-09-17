# coding: utf-8
"""Execute one explicit GPT hybrid decision; no implicit model or old IK rollout.

Use --current to forward start/observe/infer/finish to eval_control.py.
Use --session ID --decision JSON to apply a reviewed follow or correct action.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

MAX_DECISIONS = 180


def apply_decision(controller, decision):
    if controller.mode != 'hybrid':
        raise ValueError('run_hybrid requires a hybrid episode')
    action = decision.get('action')
    # Follow executes only a fresh prefix; Controller logs and invalidates suffix.
    if action == 'follow':
        return controller.follow(decision['steps'], decision['reason'])
    if action in ('correct', 'eef'):
        if 'poses' in decision:
            raise ValueError('Waypoint lists retired: supply one pose or explicit dual-arm goals')
        measured = controller.session_http('state')
        if decision.get('expected_step') != measured['t']:
            raise ValueError('Stale correction decision')
        if 'goals' in decision:
            goals = decision['goals']
        else:
            pose = decision['pose']
            if not isinstance(pose, list) or len(pose) != 4:
                raise ValueError('Single pose must be [x,y,z,gripper]')
            goals = copy.deepcopy(measured['ee'])
            side = decision.get('side', 'left')
            if side not in goals:
                raise ValueError('Unknown arm')
            goals[side]['xyz'] = pose[:3]
            goals[side]['gripper'] = pose[3]
        # The service holds ONE target and recomputes 6D IK after each actual
        # 25Hz control step, from measured joints; no qpos writes or open loop.
        return controller.eef({'expected_step': measured['t'], 'goals': goals,
                               'steps': decision['steps'], 'decision_summary': decision['reason']})
    raise ValueError('Supported decisions: follow or correct/eef; edit is not implemented')


def main():
    root = Path(os.environ.get('ASTRA_EVAL_ROOT', Path(__file__).resolve().parents[1])).expanduser()
    if not (root / 'eval_control.py').is_file():
        raise SystemExit('Set ASTRA_EVAL_ROOT to current simulation/mujoco or astra_eval directory')
    sys.path.insert(0, str(root))
    import eval_control
    if len(sys.argv) > 1 and sys.argv[1] == '--current':
        sys.argv = [str(root/'eval_control.py'), *sys.argv[2:]]
        if len(sys.argv) > 1 and sys.argv[1] == 'start' and '--mode' not in sys.argv:
            sys.argv.extend(['--mode', 'hybrid'])
        return eval_control.main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--decision', type=Path, required=True)
    args = parser.parse_args()
    controller = eval_control.Controller(args.session)
    result = apply_decision(controller, json.loads(args.decision.read_text()))
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
