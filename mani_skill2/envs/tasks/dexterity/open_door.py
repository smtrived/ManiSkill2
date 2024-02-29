from collections import OrderedDict
from typing import Union, Dict, Any, List

import numpy as np
import torch

from mani_skill2.agents.robots import XArm7Ability
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.building.articulations import build_table_door
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import get_obj_by_name
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill2.utils.structs.articulation import Articulation
from mani_skill2.utils.structs.pose import Pose
from mani_skill2.utils.structs.types import Array
from .base_env import DynamicManipulationEnv


class OpenDoorEnv(DynamicManipulationEnv):
    agent: Union[XArm7Ability]
    _clearance = 0.003

    def __init__(
            self,
            *args,
            robot_init_qpos_noise=0.02,
            valve_init_pos_noise=0.02,
            difficulty_level: int = -1,
            **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.valve_init_pos_noise = valve_init_pos_noise

        if not isinstance(difficulty_level, int) or difficulty_level >= 5 or difficulty_level < 0:
            raise ValueError(f"Difficulty level must be a int within 0-4, but get {difficulty_level}")
        self.difficulty_level = difficulty_level
        super().__init__(*args, robot_uids="xarm7_ability", **kwargs)

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

    def _load_articulations(self):
        # Robel valve
        if self.difficulty_level == 0:
            # Only tri-valve
            params = [0] * self.num_envs
        else:
            raise ValueError(f"Difficulty level must be a int within 0-4, but get {self.difficulty_level}")

        doors: List[Articulation] = []
        for i, valve_angles in enumerate(params):
            scene_mask = np.zeros(self.num_envs, dtype=bool)
            scene_mask[i] = True
            if self.difficulty_level == 0:
                door = build_table_door(self._scene, scene_mask=scene_mask, name=f"door_{i}")
            else:
                raise NotImplementedError
            doors.append(door)

        self.door = Articulation.merge(doors, "door")
        self.door_frame_links = get_obj_by_name(self.door.get_links(), "door_frame")
        self.door_body_links = get_obj_by_name(self.door.get_links(), "door_body")
        self.door_handle_links = get_obj_by_name(self.door.get_links(), "door_handle")

    def _initialize_actors(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            self.table_scene.initialize()

        # Initialize the valve
        with torch.device(self.device):
            qpos = torch.zeros((self.num_envs, 2))
            random_xy = (torch.rand((self.num_envs, 2)) * 2 - 1) * 0.05
            random_pos = torch.concat((random_xy, torch.ones((self.num_envs, 1)) * 0.01), dim=-1)
            quat = torch.zeros(self.num_envs, 4)
            quat[:, 0] = 1
            self.door.set_pose(Pose.create_from_pq(random_pos, quat))
            self.door.set_qpos(qpos)

    def _initialize_agent(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            dof = self.agent.robot.dof
            if isinstance(dof, torch.Tensor):
                dof = dof[0]
            init_qpos = torch.zeros((self.num_envs, dof))
            init_qpos += torch.randn((self.num_envs, dof)) * self.robot_init_qpos_noise
            self.agent.reset(init_qpos)
            self.agent.robot.set_pose(Pose.create_from_pq(torch.tensor([-0.6, 0, 0]), torch.tensor([1, 0, 0, 0])))

    def _get_obs_extra(self, info: Dict):
        with torch.device(self.device):
            obs = OrderedDict()
            return obs

    def evaluate(self, **kwargs) -> dict:
        return dict(success=False)

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        reward = torch.zeros(self.num_envs, device=self.device)

        # assign rewards to parallel environments that achieved success to the maximum of 3.
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4.0


@register_env("OpenDoorLevel0-v1", max_episode_steps=300)
class OpenDoorEnvLevel0(OpenDoorEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_init_qpos_noise=0.02,
                         valve_init_pos_noise=0.02, difficulty_level=0, **kwargs)
