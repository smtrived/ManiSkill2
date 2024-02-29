from copy import deepcopy
from typing import List

import sapien

from mani_skill2 import PACKAGE_ASSET_DIR
from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.controllers import *
from mani_skill2.utils.sapien_utils import get_objs_by_names


class UnitreeH1(BaseAgent):
    uid = "unitree_h1"
    urdf_path = f"/home/yuzhe/project/dexsuite/dex_urdf/robots/humaniod/unitree_h1/h1_with_hand_simple_collision.urdf"
    urdf_config = dict(
        # _materials=dict(
        #     tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
        # ),
        # link={
        #     "link_3.0_tip": dict(
        #         material="tip", patch_radius=0.1, min_patch_radius=0.1
        #     ),
        #     "link_7.0_tip": dict(
        #         material="tip", patch_radius=0.1, min_patch_radius=0.1
        #     ),
        #     "link_11.0_tip": dict(
        #         material="tip", patch_radius=0.1, min_patch_radius=0.1
        #     ),
        #     "link_15.0_tip": dict(
        #         material="tip", patch_radius=0.1, min_patch_radius=0.1
        #     ),
        # },
    )
    sensor_configs = {}

    def __init__(self, *args, **kwargs):
        self.joint_names = [
            "torso_joint",
            # leg
            "left_hip_yaw_joint",
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_joint",
            "right_hip_yaw_joint",
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_joint",
            # arm
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        ]

        self.joint_stiffness = 4e3
        self.joint_damping = 2e2
        self.joint_force_limit = 5e2

        super().__init__(*args, **kwargs)

    def _after_init(self):
        pass

    @property
    def controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        joint_pos = PDJointPosControllerConfig(
            self.joint_names,
            None,
            None,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            normalize_action=False,
        )
        joint_delta_pos = PDJointPosControllerConfig(
            self.joint_names,
            -0.1,
            0.1,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            use_delta=True,
        )
        joint_target_delta_pos = deepcopy(joint_delta_pos)
        joint_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=joint_delta_pos,
            pd_joint_pos=joint_pos,
            pd_joint_target_delta_pos=joint_target_delta_pos,
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent.
        """
        obs = super().get_proprioception()

        return obs


class UnitreeH1InspireHand(UnitreeH1):
    uid = "unitree_h1_inspire_hand"
    urdf_path = f"/home/yuzhe/project/dexsuite/dex_urdf/robots/hands/inspire_hand/h1_with_hand_simple_collision.urdf"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_names = [
            "torso_joint",
            # leg
            "left_hip_yaw_joint",
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_joint",
            "right_hip_yaw_joint",
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_joint",
            # arm
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            # hand
            "L_thumb_proximal_yaw_joint",
            "L_index_proximal_joint",
            "L_index_proximal_joint",
            "L_index_proximal_joint",
        ]

        self.joint_stiffness = 4e3
        self.joint_damping = 2e2
        self.joint_force_limit = 5e2

    def _after_init(self):
        super()._after_init()
        self.fsr_links: List[sapien.Entity] = get_objs_by_names(
            self.robot.get_links(),
            self.palm_fsr_link_names + self.finger_fsr_link_names,
        )
