# coding: utf-8
"""Explicit gateway; legacy payloads and implicit model selection are retired."""
import os
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    entry = Path(__file__).stem
    mode = 'direct' if entry == 'run_direct' else 'pi05' if entry == 'run_pi05_only' else 'hybrid'
    root = Path(os.environ['ASTRA_EVAL_ROOT']).expanduser() if os.environ.get('ASTRA_EVAL_ROOT') else Path(__file__).resolve().parents[1]
    current = root / 'eval_control.py'
    if len(sys.argv) < 2 or sys.argv[1] != '--current':
        raise SystemExit('Old runner retired: it used obsolete ports/protocol and implicit Qwen. '
                         'Use eval_control.py in simulation/mujoco. Optional explicit gateway: '
                         f'ASTRA_EVAL_ROOT=/path/to/simulation/mujoco python3 {entry}.py --current start --mode {mode} --layout 0. '
                         'This starts an episode only; a fresh GPT context must perform observe/infer/follow/eef/finish.')
    if not current.is_file():
        raise SystemExit('Set ASTRA_EVAL_ROOT to the directory containing current eval_control.py')
    args = sys.argv[2:]
    if args and args[0] == 'start' and '--mode' not in args:
        args += ['--mode', mode]
    sys.argv = [str(current), *args]
    sys.path.insert(0, str(root))
    runpy.run_path(str(current), run_name='__main__')
