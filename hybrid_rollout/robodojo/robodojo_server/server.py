"""Launch RoboDojo's native evaluation environment behind the existing RPC."""
import argparse
import faulthandler
import importlib
import json
import os
import signal
import shutil
import traceback
from pathlib import Path


class NoPolicyConnection:
    """Native reset callback only; inference is owned by the Codex tool host."""
    def __init__(self, **kwargs):
        pass

    def call(self, func_name, **kwargs):
        if func_name != 'reset':
            raise RuntimeError('Native autonomous policy loop is disabled')

    def close(self):
        pass


def main():
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    import cv2  # Load before Isaac's extension dependency paths.
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', default='build_tower')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=19113)
    p.add_argument('--eval-seed', type=int, default=0, help='Published layout group, NOT the reset layout ID')
    p.add_argument('--eval-manifest', type=Path)
    p.add_argument('--case-file', type=Path)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args(); args.headless = True; args.enable_cameras = True
    from ..evaluation import read_panel, verify_assets, case_identity
    from ..io import require
    from env.global_configs import ROOT_DIR, BENCHMARK
    registry = importlib.import_module(f'task.{BENCHMARK}.task_registry')
    evaluation_case = None
    if args.eval_manifest:
        panel = read_panel(args.eval_manifest, os.environ.get('ROLLOUT_EVAL_MANIFEST_SHA256'))
        selected = json.loads(args.case_file.read_text())
        evaluation_case = selected['case']
        require(evaluation_case in panel['cases'] and selected['identity'] == case_identity(panel, evaluation_case),
                'Selected case differs from the frozen panel')
        require((args.task, args.eval_seed) == (evaluation_case['runtime_task'], evaluation_case['eval_seed']),
                'CLI task/eval_seed differs from selected case')
        verify_assets(panel, ROOT_DIR, [evaluation_case])
        args.output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(ROOT_DIR)/evaluation_case['layout']['path'], args.output/'scene_layout.json')
    # Match native main.py: monitor Articulation tasks before AppLauncher.
    import yaml
    task_config_path = registry.task_config_path(str(Path(ROOT_DIR)/'task'/BENCHMARK/'config'), args.task)
    with open(task_config_path) as stream:
        task_values = yaml.safe_load(stream) or {}
    enable_monitor = bool(task_values.get('Articulation'))
    if enable_monitor:
        from src.eval_client.physx_warning_monitor import get_monitor
        get_monitor().start(enabled=True)
    app = AppLauncher(args).app
    env = session = None
    try:
        from omegaconf import OmegaConf
        from env.global_configs import ENV_CONFIG_PATH, ROOT_DIR, BENCHMARK
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization
        from src.eval_client import eval_env
        # RoboDojo's GeometryObject.get_state builds its pose from USD on the
        # CPU, but keeps env_origin on the CUDA simulation device.  The native
        # reward checks subtract those tensors directly and fail on the first
        # action with a CPU/CUDA device mismatch.  Keep the compatibility fix
        # in this adapter rather than modifying the checked-out RoboDojo tree.
        from env.scene_manager.objects.geometry import GeometryObject
        native_geometry_get_state = GeometryObject.get_state

        def geometry_get_state_device_safe(geometry, *call_args, **call_kwargs):
            origin = getattr(geometry, 'env_origin', None)
            if hasattr(origin, 'detach'):
                geometry.env_origin = origin.detach().cpu()
            try:
                return native_geometry_get_state(geometry, *call_args, **call_kwargs)
            finally:
                if origin is not None:
                    geometry.env_origin = origin

        GeometryObject.get_state = geometry_get_state_device_safe
        from .rpc import serve
        from .session import RoboDojoSession
        root = Path(ENV_CONFIG_PATH)
        evaluation = load_yaml(str(root/'arx_x5.yml'))
        evaluation.update(task_name=args.task, num_envs=1, device_id=0, eval_batch=False,
            policy_name='Pi_05', additional_info='codex_hybrid', seed=args.eval_seed,
            physx_monitor_enabled=enable_monitor)
        values = {key: load_yaml(str(root/key/(evaluation['config'][key]+'.yml')))
                  for key in ('sim', 'scene', 'camera', 'robot')}
        # Monitoring-only overview camera. GPT policy inputs remain the three
        # native cameras declared by RoboDojoSession.CAMERAS.
        values['camera']['cam_monitor_overview'] = dict(camera=dict(
            type='third_view', mesh='pinhole', pos=[0.0, -1.60, 1.70],
            ori=[58.0, 0.0, 0.0]))
        values.update(eval_cfg=evaluation, deploy_cfg=dict(port=1, policy_name='Pi_05'),
            task_env=load_yaml(registry.task_config_path(str(Path(ROOT_DIR)/'task'/BENCHMARK/'config'), args.task)))
        cfg = OmegaConf.create(values)
        cfg.sim.scene.num_envs = 1
        # AppLauncher selecting CUDA does not rewrite RoboDojo's nested
        # simulation config. Keep physics and rendering on the requested GPU;
        # the stock YAML otherwise leaves this headless process on CPU and its
        # first RTX render can block indefinitely.
        cfg.sim.device = args.device
        cfg = process_randomization(cfg)
        cfg, _ = process_config(cfg, task_name=args.task)
        cfg.eval_cfg.eval_num = 1
        cfg.camera.default_frequency = cfg.eval_cfg.observation.collect_freq
        cfg.sim.seed = [0]  # Native startup; the controller later resets the selected layout_id.
        # EEF corrections use bounded robot-only DLS, never object-aware cuRobo.
        for robot in cfg.robot.robots:
            robot.need_planner = False
        original = eval_env.WsModelClient
        try:
            eval_env.WsModelClient = NoPolicyConnection
            env = eval_env.create_eval_env(cfg, app)
        finally:
            eval_env.WsModelClient = original
        # Reuse the headless monitor lighting that the validated direct GPU
        # runner installs before its first render.  Installing it lazily is
        # necessary because EvalEnv creates the USD stage during reset.
        native_render = env.render
        monitor_light_ready = False

        def render_with_monitor_light():
            nonlocal monitor_light_ready
            if not monitor_light_ready:
                from pxr import Gf, UsdLux
                from isaacsim.core.utils.stage import get_current_stage
                light = UsdLux.DomeLight.Define(get_current_stage(), '/World/monitor_fill_light')
                light.GetIntensityAttr().Set(250.0)
                light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.95, 0.90))
                monitor_light_ready = True
            return native_render()

        env.render = render_with_monitor_light
        if evaluation_case:
            resolved = env.seed_manager.seed_info[evaluation_case['layout_id']]['scene_layout']
            require(Path(resolved).resolve() == (Path(ROOT_DIR)/evaluation_case['layout']['path']).resolve(),
                    'Native SeedManager resolved a different layout')
        from ..io import write_json
        write_json(args.output/'resolved_config.json', OmegaConf.to_container(cfg, resolve=True))
        write_json(args.output/'adapter_runtime.json', dict(
            schema='robodojo_rollout.adapter_runtime.v1',
            task=args.task,
            layout_id=args.eval_seed,
            simulator_device=str(cfg.sim.device),
            physics_dt=float(cfg.sim.dt),
            control_frequency_hz=int(cfg.eval_cfg.observation.collect_freq),
            policy_cameras=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
            monitor_camera='cam_monitor_overview',
            direct_rgb_capture=os.environ.get('ROBODOJO_DIRECT_RGB_CAPTURE') == '1',
            skip_duplicate_scene_setup=os.environ.get('ROBODOJO_SKIP_EVAL_SCENE_SETUP') == '1',
            post_reset_warmup_renders=int(os.environ.get('ROBODOJO_POST_RESET_WARMUP_RENDERS', '3')),
            monitor_fill_light=dict(intensity=250.0, color=[1.0, 0.95, 0.90]),
            compatibility_fixes=['geometry_env_origin_cpu_cuda_alignment']))
        session = RoboDojoSession(env, args.output, args.task,
            evaluation_identity=selected['identity'] if evaluation_case else None)
        serve(session, args.port)
    except BaseException:
        # SimulationApp.close may exit before Python prints an uncaught error.
        # Persist the original failure before native cleanup can obscure it.
        traceback.print_exc()
        raise
    finally:
        if session:
            session._write_summary('server_close')
        if env:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
