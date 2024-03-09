import argparse
from ast import parse
import gymnasium as gym
import numpy as np
import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv

from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
import sapien.utils.viewer
import h5py
import json
import mani_skill.trajectory.utils as trajectory_utils
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.examples.teleoperation.space_mouse import SpaceMouseThread


def main(args):
    output_dir = f"{args.record_dir}/teleop/{args.env_id}"
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        reward_mode="sparse",
        # shader_dir="rt-fast",
    )
    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=False,
        info_on_video=False,
        source_type="teleoperation",
        source_desc="teleoperation via the click+drag system"
    )
    num_trajs = 0
    seed = 0
    device = SpaceMouseThread()

    env.reset(seed=seed)
    while True:
        print(f"Collecting trajectory {num_trajs+1}, seed={seed}")
        code = solve(env, device=device, vis=True)
        if code == "quit":
            num_trajs += 1
            break
        elif code == "continue":
            seed += 1
            num_trajs += 1
            env.reset(seed=seed)
            continue
        elif code == "restart":
            env.reset(seed=seed, options=dict(save_trajectory=False))
    h5_file_path = env._h5_file.filename
    json_file_path = env._json_path
    env.close()
    del env
    # print(f"saving videos to {output_dir}")

    # trajectory_data = h5py.File(h5_file_path)
    # with open(json_file_path, "r") as f:
    #     json_data = json.load(f)
    # env = gym.make(
    #     args.env_id,
    #     obs_mode=args.obs_mode,
    #     control_mode="pd_joint_pos",
    #     render_mode="rgb_array",
    #     reward_mode="sparse",
    #     # shader_dir="rt",
    # )
    # env = RecordEpisode(
    #     env,
    #     output_dir=output_dir,
    #     trajectory_name="trajectory",
    #     save_video=True,
    #     info_on_video=False,
    #     save_trajectory=False,
    #     video_fps=30
    # )
    # for episode in json_data["episodes"]:
    #     traj_id = f"traj_{episode['episode_id']}"
    #     data = trajectory_data[traj_id]
    #     env.reset(**episode["reset_kwargs"])
    #     env_states_list = trajectory_utils.dict_to_list_of_dicts(data["env_states"])

    #     env.base_env.set_state_dict(env_states_list[0])
    #     for action in np.array(data["actions"]):
    #         env.step(action)

    # trajectory_data.close()
    # env.close()
    # del env



def solve(env: BaseEnv, device: SpaceMouseThread, vis=False):
    while True:
        env.render_human()
        action_dict = {
            "lb": device.get_left_button(),
            "rb": device.get_right_button(),
            "xyz": device.get_xyz(),
            "rpy": device.get_rpy()
        }
        d_xyz = device.get_xyz()
        print("d_xyz", d_xyz)
        d_rpy = device.get_rpy()
        # d_rpy = np.array([0, 0, 1])
        print("d_rpy", d_rpy)

        # env.step(action)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1")
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument("-r", "--robot-uid", type=str, default="panda", help="Robot setups supported are ['panda']")
    parser.add_argument("--record-dir", type=str, default="demos")
    args, opts = parser.parse_known_args()

    return args
if __name__ == "__main__":
    main(parse_args())
