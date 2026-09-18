"""Exercise the deployed policy factory without loading 6.8GB of weights."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class PolicyContractTest(unittest.TestCase):
    def build_config(self):
        tree = ast.parse(
            Path(__file__).with_name("pi05_service.py").read_text(encoding="utf-8")
        )
        factory = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "build_policy"
        )

        # Generic openpi defaults deliberately reproduce the deployment bug.
        def aloha(**kwargs):
            return SimpleNamespace(
                **{"adapt_to_pi": True, "use_delta_joint_actions": True, **kwargs}
            )

        creator = Mock()
        scope = dict(
            train_config_mod=SimpleNamespace(
                LeRobotAlohaDataConfig=aloha,
                AssetsConfig=SimpleNamespace,
                DataConfig=SimpleNamespace,
                TrainConfig=SimpleNamespace,
            ),
            pi0_config=SimpleNamespace(Pi0Config=SimpleNamespace),
            ocp_checkpoints=SimpleNamespace(load_norm_stats=Mock(return_value={})),
            transforms=SimpleNamespace(Group=Mock(), RepackTransform=Mock()),
            policy_config=SimpleNamespace(create_trained_policy=creator),
            pathlib=__import__("pathlib"),
            ASSETS_DIR="assets",
            ASSET_ID="arx_x5_sim",
            CKPT_DIR="weights",
            DEVICE="npu",
            CAM_KEYS=("cam_high", "cam_left_wrist", "cam_right_wrist"),
        )
        exec(
            compile(
                ast.Module(body=[factory], type_ignores=[]), "policy_factory", "exec"
            ),
            scope,
        )
        scope["build_policy"]()
        return creator.call_args.args[0].data

    def test_arx_does_not_apply_aloha_conversion(self):
        self.assertIs(self.build_config().adapt_to_pi, False)

    def test_delta_joint_actions_are_retained(self):
        self.assertIs(self.build_config().use_delta_joint_actions, True)


if __name__ == "__main__":
    unittest.main()
