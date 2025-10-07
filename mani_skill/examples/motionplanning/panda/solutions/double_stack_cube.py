import argparse
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.double_stack_cube import DoubleStackCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.wrappers.record import RecordEpisode

def solve(env: DoubleStackCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    def get_grasp_pose(actor):
        obb = get_actor_obb(actor)
        approaching = np.array([0, 0, -1])
        target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=FINGER_LENGTH
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        base_grasp = env.agent.build_grasp_pose(approaching, closing, center)

        # search a valid pose
        angles = np.array([0, np.pi / 2, np.pi / 2 * 3, np.pi])
        grasp_pose = base_grasp
        for angle in angles:
            delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
            grasp_pose2 = base_grasp * delta_pose
            res = planner.move_to_pose_with_RRTConnect(grasp_pose2, dry_run=True)
            if res == -1:
                continue
            grasp_pose = grasp_pose2
            break
        else:
            print("Fail to find a valid grasp pose")
        return grasp_pose

    def pick(grasp_pose):
        # reach
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        planner.move_to_pose_with_RRTConnect(reach_pose)

        # grasp
        planner.move_to_pose_with_RRTConnect(grasp_pose)
        planner.close_gripper()

        # lift
        lift_pose = sapien.Pose([0, 0, 0.15]) * grasp_pose
        planner.move_to_pose_with_RRTConnect(lift_pose)
        return lift_pose

    def stack(target_actor, lift_pose, top_actor):
        # stack
        goal_pose = target_actor.pose * sapien.Pose([0, 0, (env.cube_half_size[2] * 2).item()])
        offset = (goal_pose.p - top_actor.pose.p).cpu().numpy()[0]
        align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
        planner.move_to_pose_with_RRTConnect(align_pose)
        res = planner.open_gripper()

        # lift
        lift_pose = sapien.Pose([0, 0, 0.15]) * align_pose
        planner.move_to_pose_with_RRTConnect(lift_pose)
        return align_pose, res

    # --- Pick A (red) and place on B (green) ---
    grasp_pose = get_grasp_pose(env.cubeA)
    lift_pose = pick(grasp_pose)
    align_pose, _ = stack(env.cubeB, lift_pose, env.cubeA)

    # --- Pick C (blue) and place on A (red) ---
    grasp_pose = get_grasp_pose(env.cubeC)
    lift_pose = pick(grasp_pose)
    _, res = stack(env.cubeA, lift_pose, env.cubeC)

    planner.close()
    return res


