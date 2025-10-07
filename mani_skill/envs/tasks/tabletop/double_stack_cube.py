from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
import os

@register_env("DoubleStackCube-v1", max_episode_steps=50)
class DoubleStackCubeEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to first pick the red cube and place it on the green cube,
    then pick the blue cube and place it on the red cube and let go of the cube without it falling

    **Randomizations:**
    - 3 cubes have their z-axis rotation randomized
    - 3 cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other

    **Success Conditions:**
    - the red cube is on top of the green cube (to within half of the cube size)
    - the blue cube is on top of the red cube (to within half of the cube size)
    - blue and red cube are static
    - the blue cube is not being grasped by the robot (robot must let go of the cube)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/DoubleStackCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 1000, 1000, 1, 0.01, 100)  # Update the human rendering camera to generate images with 1000 x 1000 pixels

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        # red cube
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[1, 0, 0, 1],
            name="cubeA",
            add_collision=True,
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        # green cube
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="cubeB",
            add_collision=True,
            initial_pose=sapien.Pose(p=[1, 0, 0.1]),
        )
        # blue cube
        self.cubeC = actors.build_cube(
            self.scene,
            half_size=0.02, # 4 cm
            color=[0, 0, 1, 1],
            name="cubeC",
            add_collision=True,
            initial_pose=sapien.Pose(p=[0.5, 0.5, 0.1]),
        )

    def _load_lighting(self, options: dict):
        shadow = self.enable_shadow
        self.scene.set_ambient_light([0.5, 0.5, 0.0]) # change the ambient light so that the scene is slightly yellow
        self.scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=shadow, shadow_scale=5, shadow_map_size=2048
        )
        self.scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2)) * 0.05
            region = [[-0.2, -0.2], [0.2, 0.2]]  # 40cm by 40cm wide square centered at the world origin
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            cubeA_xy = sampler.sample(radius, 100)
            cubeB_xy = sampler.sample(radius, 100, verbose=False)
            cubeC_xy = sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            xyz[:, :2] = cubeC_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeC.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def evaluate(self):
        pos_A = self.cubeA.pose.p  # red
        pos_B = self.cubeB.pose.p  # green
        pos_C = self.cubeC.pose.p  # blue

        xy_tol = torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        z_step = self.cube_half_size[..., 2] * 2
        z_tol = 0.005

        def is_on(top_pos, bottom_pos):
            offset = top_pos - bottom_pos
            xy_ok = torch.linalg.norm(offset[..., :2], axis=1) <= xy_tol
            z_ok = torch.abs(offset[..., 2] - z_step) <= z_tol
            return torch.logical_and(xy_ok, z_ok)

        is_A_on_B = is_on(pos_A, pos_B)  # red on green
        is_C_on_A = is_on(pos_C, pos_A)  # blue on red

        # check whether blue and red blocks are static
        is_A_static = self.cubeA.is_static(lin_thresh=3e-2, ang_thresh=0.5)
        is_C_static = self.cubeC.is_static(lin_thresh=3e-2, ang_thresh=0.5)

        # check whether A&C not grasped
        is_C_grasped = self.agent.is_grasping(self.cubeC)
        is_A_grasped = self.agent.is_grasping(self.cubeA)

        success = is_A_on_B * is_C_on_A * (~is_C_grasped) * (~is_A_grasped) * is_A_static * is_C_static

        return {
            "is_A_on_B": is_A_on_B,
            "is_C_on_A": is_C_on_A,
            "is_A_static": is_A_static,
            "is_C_static": is_C_static,
            "is_A_grasped": is_A_grasped,
            "is_C_grasped": is_C_grasped,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                cubeC_pose=self.cubeC.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeC_pos=self.cubeC.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
                cubeC_to_cubeB_pos=self.cubeB.pose.p - self.cubeC.pose.p,  # blue - green
                cubeC_to_cubeA_pos=self.cubeA.pose.p - self.cubeC.pose.p,  # blue - red
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp = self.agent.tcp.pose.p
        A = self.cubeA.pose.p
        B = self.cubeB.pose.p
        C = self.cubeC.pose.p

        z_step = self.cube_half_size[2] * 2
        goal_A = torch.hstack([B[:, :2], (B[:, 2] + z_step)[:, None]])
        goal_C = torch.hstack([A[:, :2], (A[:, 2] + z_step)[:, None]])

        is_A_on_B = info.get("is_cubeA_on_cubeB", torch.zeros(A.shape[0], dtype=torch.bool, device=self.device))
        is_C_on_A = info.get("is_cubeC_on_cubeA", torch.zeros(A.shape[0], dtype=torch.bool, device=self.device))
        phase1 = is_A_on_B

        # reaching reward (to A before phase1, to C after)
        dist_A = torch.linalg.norm(tcp - A, axis=1)
        dist_C = torch.linalg.norm(tcp - C, axis=1)
        reach_dist = torch.where(~phase1, dist_A, dist_C)
        reward = 2 * (1 - torch.tanh(5.0 * reach_dist))

        # grasp and place reward
        is_A_grasped = info["is_A_grasped"]
        is_C_grasped = info["is_C_grasped"]

        place_A_dist = torch.linalg.norm(goal_A - A, axis=1)
        place_C_dist = torch.linalg.norm(goal_C - C, axis=1)
        place_A = 1 - torch.tanh(5.0 * place_A_dist)
        place_C = 1 - torch.tanh(5.0 * place_C_dist)

        mask0 = (~phase1) & is_A_grasped
        mask1 = phase1 & is_C_grasped
        tmp = reward.clone()
        tmp[mask0] = (4 + place_A)[mask0]
        tmp[mask1] = (4 + place_C)[mask1]
        reward = tmp

        # ungrasp and static reward (per phase, same 6 + avg(...) pattern)
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)

        ungrasp_A = torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        ungrasp_A[~is_A_grasped] = 1.0
        vA = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        wA = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_A = 1 - torch.tanh(vA * 10 + wA)

        ungrasp_C = torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        ungrasp_C[~is_C_grasped] = 1.0
        vC = torch.linalg.norm(self.cubeC.linear_velocity, axis=1)
        wC = torch.linalg.norm(self.cubeC.angular_velocity, axis=1)
        static_C = 1 - torch.tanh(vC * 10 + wC)

        tmp = reward.clone()
        tmp[is_A_on_B] = (6 + (ungrasp_A + static_A) / 2.0)[is_A_on_B]
        tmp[is_C_on_A & phase1] = (6 + (ungrasp_C + static_C) / 2.0)[is_C_on_A & phase1]
        reward = tmp

        success = info.get("success", torch.zeros(A.shape[0], dtype=torch.bool, device=self.device))
        reward[success] = 8.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8.0

