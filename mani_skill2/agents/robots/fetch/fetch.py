from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill2 import PACKAGE_ASSET_DIR
from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.controllers import *
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.common import compute_angle_between, np_compute_angle_between
from mani_skill2.utils.sapien_utils import (
    compute_total_impulse,
    get_actor_contacts,
    get_obj_by_name,
    get_pairwise_contact_impulse,
    to_tensor,
)
from mani_skill2.utils.structs.actor import Actor
from mani_skill2.utils.structs.base import BaseStruct
from mani_skill2.utils.structs.joint import Joint
from mani_skill2.utils.structs.link import Link
from mani_skill2.utils.structs.pose import Pose
from mani_skill2.utils.structs.types import Array

FETCH_UNIQUE_COLLISION_BIT = 1 << 30


class Fetch(BaseAgent):
    uid = "fetch"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/fetch/fetch.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            fetch_leftfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            fetch_rightfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )
    sensor_configs = [
        CameraConfig(
            uid="fetch_head",
            p=[0, 0, 0],
            q=[1, 0, 0, 0],
            width=128,
            height=128,
            fov=2,
            near=0.01,
            far=10,
            entity_uid="head_camera_link",
        )
    ]
    REACHABLE_DIST = 1.5

    def __init__(self, *args, **kwargs):
        self.arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100

        self.gripper_joint_names = [
            "l_gripper_finger_joint",
            "r_gripper_finger_joint",
        ]
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100

        self.ee_link_name = "gripper_link"

        self.body_joint_names = [
            "head_pan_joint",
            "head_tilt_joint",
            "torso_lift_joint",
        ]
        self.body_stiffness = 1e3
        self.body_damping = 1e2
        self.body_force_limit = 100

        self.base_joint_names = [
            "root_x_axis_joint",
            "root_y_axis_joint",
            "root_z_rotation_joint",
        ]

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
            normalize_action=True,
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
            0.05,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        )

        # -------------------------------------------------------------------------- #
        # Body
        # -------------------------------------------------------------------------- #
        body_pd_joint_pos = PDJointPosControllerConfig(
            self.body_joint_names,
            [-1.57, -1.6056, 0],
            [1.57, 1.6056, 0.38615],
            self.body_stiffness,
            self.body_damping,
            self.body_force_limit,
            normalize_action=True,
        )

        # -------------------------------------------------------------------------- #
        # Base
        # -------------------------------------------------------------------------- #
        base_pd_joint_vel = PDBaseVelControllerConfig(
            self.base_joint_names,
            lower=[-0.5, -0.5, -3.14],
            upper=[0.5, 0.5, 3.14],
            damping=1000,
            force_limit=500,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_delta_pose_align=dict(
                arm=arm_pd_ee_delta_pose_align,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(
                arm=arm_pd_joint_vel,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.finger1_link: Link = get_obj_by_name(
            self.robot.get_links(), "l_gripper_finger_link"
        )
        self.finger2_link: Link = get_obj_by_name(
            self.robot.get_links(), "r_gripper_finger_link"
        )
        self.tcp: Link = get_obj_by_name(self.robot.get_links(), self.ee_link_name)

        self.base_link: Link = get_obj_by_name(self.robot.get_links(), "base_link")
        self.l_wheel_link: Link = get_obj_by_name(
            self.robot.get_links(), "l_wheel_link"
        )
        self.r_wheel_link: Link = get_obj_by_name(
            self.robot.get_links(), "r_wheel_link"
        )
        for link in [self.l_wheel_link, self.r_wheel_link]:
            for body in link._bodies:
                cs = body.get_collision_shapes()[0]
                cg = cs.get_collision_groups()
                cg[2] |= FETCH_UNIQUE_COLLISION_BIT
                cs.set_collision_groups(cg)

        self.torso_lift_link: Link = get_obj_by_name(
            self.robot.get_links(), "torso_lift_link"
        )

        self.torso_lift_link: Link = get_obj_by_name(
            self.robot.get_links(), "torso_lift_link"
        )

        self.queries: Dict[str, Tuple[physx.PhysxGpuContactQuery, Tuple[int]]] = dict()

    def is_grasping(self, object: Actor = None, min_impulse=1e-6, max_angle=85):
        # TODO (stao): is_grasping code needs to be updated for new GPU sim
        if physx.is_gpu_enabled():
            if object.name not in self.queries:
                body_pairs = list(zip(self.finger1_link._bodies, object._bodies))
                body_pairs += list(zip(self.finger2_link._bodies, object._bodies))
                self.queries[object.name] = (
                    self.scene.px.gpu_create_contact_query(body_pairs),
                    (len(object._bodies), 3),
                )
            query, contacts_shape = self.queries[object.name]
            self.scene.px.gpu_query_contacts(query)
            # query.cuda_contacts # (num_unique_pairs * num_envs, 3)
            contacts = (
                query.cuda_contacts.torch().clone().reshape((-1, *contacts_shape))
            )
            lforce = torch.linalg.norm(contacts[0], axis=1)
            rforce = torch.linalg.norm(contacts[1], axis=1)

            # NOTE (stao): 0.5 * time_step is a decent value when tested on a pick cube task.
            min_force = 0.5 * self.scene.px.timestep

            # direction to open the gripper
            ldirection = -self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
            rdirection = self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
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
                finger1_contacts = get_actor_contacts(
                    contacts, self.finger1_link._bodies[0].entity
                )
                finger2_contacts = get_actor_contacts(
                    contacts, self.finger2_link._bodies[0].entity
                )
                return (
                    np.linalg.norm(compute_total_impulse(finger1_contacts))
                    >= min_impulse
                    and np.linalg.norm(compute_total_impulse(finger2_contacts))
                    >= min_impulse
                )
            else:
                limpulse = get_pairwise_contact_impulse(
                    contacts,
                    self.finger1_link._bodies[0].entity,
                    object._bodies[0].entity,
                )
                rimpulse = get_pairwise_contact_impulse(
                    contacts,
                    self.finger2_link._bodies[0].entity,
                    object._bodies[0].entity,
                )

                # direction to open the gripper
                ldirection = -self.finger1_link.pose.to_transformation_matrix()[
                    ..., :3, 1
                ]
                rdirection = self.finger2_link.pose.to_transformation_matrix()[
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

                return torch.Tensor([all([lflag, rflag])])

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

    @property
    def tcp_pose(self) -> Pose:
        p = (self.finger1_link.pose.p + self.finger2_link.pose.p) / 2
        q = (self.finger1_link.pose.q + self.finger2_link.pose.q) / 2
        return Pose.create_from_pq(p=p, q=q)
