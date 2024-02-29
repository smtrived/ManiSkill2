import sapien.core as sapien
from collections import OrderedDict
from typing import Union, Dict, Any, List

import numpy as np
import torch

from mani_skill2.agents.robots import UnitreeH1
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.building.articulations import build_robel_valve
from mani_skill2.utils.geometry.rotation_conversions import axis_angle_to_quaternion
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import get_obj_by_name
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill2.utils.structs.articulation import Articulation
from mani_skill2.utils.structs.pose import Pose
from mani_skill2.utils.structs.types import Array


class HumanoidEnv(BaseEnv):
    agent: Union[UnitreeH1]
    _clearance = 0.003

    def __init__(
        self,
        *args,
        robot_init_qpos_noise=0.02,
        valve_init_pos_noise=0.02,
        difficulty_level: int = -1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.valve_init_pos_noise = valve_init_pos_noise

        if (
            not isinstance(difficulty_level, int)
            or difficulty_level >= 5
            or difficulty_level < 0
        ):
            raise ValueError(
                f"Difficulty level must be a int within 0-4, but get {difficulty_level}"
            )
        self.difficulty_level = difficulty_level

        super().__init__(*args, robot_uids="unitree-h1", **kwargs)

    def _register_sensors(self):
        pose = look_at(eye=[0.3, 0, 0.3], target=[-0.1, 0, 0.05])
        return [
            CameraConfig("base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10)
        ]

    def _register_human_render_cameras(self):
        pose = look_at([0.2, 0.4, 0.4], [0.0, 0.0, 0.1])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _load_actors(self):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.table_scene.table.set_pose(sapien.Pose([0, 0, -10]))

    def _load_articulations(self):
        pass

    def _initialize_actors(self, env_idx: torch.Tensor):
        pass

    def _initialize_agent(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            dof = self.agent.robot.dof
            if isinstance(dof, torch.Tensor):
                dof = dof[0]

            init_qpos = torch.randn((b, dof)) * self.robot_init_qpos_noise
            self.agent.reset(init_qpos)
            self.agent.robot.set_pose(
                Pose.create_from_pq(
                    torch.tensor([0.0, 0, 0.28]), torch.tensor([1, 0, 0, 0])
                )
            )

    def _get_obs_extra(self, info: Dict):
        with torch.device(self.device):
            obs = OrderedDict()
            return obs

    def evaluate(self, **kwargs) -> dict:
        with torch.device(self.device):
            return dict(success=torch.zeros([self.num_envs]))

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        with torch.device(self.device):
            return torch.zeros([self.num_envs])

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 1.0


@register_env("HumanoidEnvLevel0-v1", max_episode_steps=300)
class HumanoidEnvLevel0(HumanoidEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            valve_init_pos_noise=0.02,
            difficulty_level=4,
            **kwargs,
        )
