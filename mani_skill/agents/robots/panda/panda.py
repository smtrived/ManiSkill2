from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.common import compute_angle_between, np_compute_angle_between
from mani_skill.utils.structs.actor import Actor


@register_agent()
class Panda(BaseAgent):
    uid = "panda"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v2.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            panda_leftfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            panda_rightfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    def __init__(self, *args, **kwargs):
        self.arm_joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100

        self.gripper_joint_names = [
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100

        self.ee_link_name = "panda_hand_tcp"

        super().__init__(*args, **kwargs)

    @property
    def controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD ee position (for human-interaction/teleoperation)
        arm_pd_ee_delta_pose_align = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_delta_pose_align.frame = "ee_align"

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -0.01,  # a trick to have force when the object is thin
            0.04,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos
            ),
            pd_ee_delta_pose_align=dict(
                arm=arm_pd_ee_delta_pose_align, gripper=gripper_pd_joint_pos
            ),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(arm=arm_pd_joint_vel, gripper=gripper_pd_joint_pos),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel, gripper=gripper_pd_joint_pos
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel, gripper=gripper_pd_joint_pos
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    sensor_configs = []

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_leftfinger"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_rightfinger"
        )
        self.finger1pad_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_leftfinger_pad"
        )
        self.finger2pad_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_rightfinger_pad"
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        self.queries: Dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
        ] = dict()

    def is_grasping(self, object: Actor = None, min_impulse=1e-6, max_angle=85):
        if physx.is_gpu_enabled():
            if object.name not in self.queries:
                body_pairs = list(zip(self.finger1_link._bodies, object._bodies))
                body_pairs += list(zip(self.finger2_link._bodies, object._bodies))
                self.queries[object.name] = (
                    self.scene.px.gpu_create_contact_pair_impulse_query(body_pairs),
                    (len(object._bodies), 3),
                )
            query, contacts_shape = self.queries[object.name]
            self.scene.px.gpu_query_contact_pair_impulses(query)
            # query.cuda_contacts # (num_unique_pairs * num_envs, 3)
            contacts = (
                query.cuda_impulses.torch().clone().reshape((-1, *contacts_shape))
            )
            lforce = torch.linalg.norm(contacts[0], axis=1)
            rforce = torch.linalg.norm(contacts[1], axis=1)

            # NOTE (stao): 0.5 * time_step is a decent value when tested on a pick cube task.
            min_force = 0.5 * self.scene.px.timestep

            # direction to open the gripper
            ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
            rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
            langle = compute_angle_between(ldirection, contacts[0])
            rangle = compute_angle_between(rdirection, contacts[1])
            lflag = torch.logical_and(
                lforce >= min_force, torch.rad2deg(langle) <= max_angle
            )
            rflag = torch.logical_and(
                rforce >= min_force, torch.rad2deg(rangle) <= max_angle
            )

            return torch.logical_and(lflag, rflag)
        else:
            contacts = self.scene.get_contacts()

            if object is None:
                finger1_contacts = sapien_utils.get_actor_contacts(
                    contacts, self.finger1_link
                )
                finger2_contacts = sapien_utils.get_actor_contacts(
                    contacts, self.finger2_link
                )
                return (
                    np.linalg.norm(sapien_utils.compute_total_impulse(finger1_contacts))
                    >= min_impulse
                    and np.linalg.norm(
                        sapien_utils.compute_total_impulse(finger2_contacts)
                    )
                    >= min_impulse
                )
            else:
                limpulse = sapien_utils.get_pairwise_contact_impulse(
                    contacts, self.finger1_link, object
                )
                rimpulse = sapien_utils.get_pairwise_contact_impulse(
                    contacts, self.finger2_link, object
                )

                # direction to open the gripper
                ldirection = self.finger1_link.pose.to_transformation_matrix()[
                    ..., :3, 1
                ]
                rdirection = -self.finger2_link.pose.to_transformation_matrix()[
                    ..., :3, 1
                ]

                # TODO Convert this to batched code
                # angle between impulse and open direction
                langle = np_compute_angle_between(ldirection[0], limpulse)
                rangle = np_compute_angle_between(rdirection[0], rimpulse)

                lflag = (
                    np.linalg.norm(limpulse) >= min_impulse
                    and np.rad2deg(langle) <= max_angle
                )
                rflag = (
                    np.linalg.norm(rimpulse) >= min_impulse
                    and np.rad2deg(rangle) <= max_angle
                )

                return all([lflag, rflag])

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (panda_hand_tcp)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    # sensor_configs = [
    #     CameraConfig(
    #         uid="hand_camera",
    #         p=[0.0464982, -0.0200011, 0.0360011],
    #         q=[0, 0.70710678, 0, 0.70710678],
    #         width=128,
    #         height=128,
    #         fov=1.57,
    #         near=0.01,
    #         far=100,
    #         entity_uid="panda_hand",
    #     )
    # ]
