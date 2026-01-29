# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    JointPositionActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.utils import configclass

from uwlab.controllers.differential_ik_cfg import MultiConstraintDifferentialIKControllerCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from uwlab.envs.mdp.actions.actions_cfg import (
    DefaultJointPositionStaticActionCfg,
    MultiConstraintsDifferentialInverseKinematicsActionCfg,
    TransformedOneShotDifferentialIKActionCfg,
)
import torch
import isaaclab.utils.math as math_utils

from uwlab_tasks.manager_based.manipulation.reset_states.mdp.utils import read_metadata_from_usd_directory
from uwlab_assets.robots.ur5e_robotiq_gripper import EXPLICIT_UR5E_ROBOTIQ_2F85
from uwlab_tasks.manager_based.manipulation.reset_states.mdp.actions.actions_cfg import TransformedOperationalSpaceControllerActionCfg

"""
UR5E ROBOTIQ 2F85 ACTIONS
"""
DROID_JOINT_POSITION: JointPositionActionCfg = JointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    scale=1.0,
    use_default_offset=False,
)

DROID_RELATIVE_JOINT_POSITION: RelativeJointPositionActionCfg = RelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    scale=0.02,
    # scale=0.02,
    use_zero_offset=True,
)

DROID_MC_IKABSOLUTE_ARM = MultiConstraintsDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    body_name=["robotiq_base_link"],
    controller=MultiConstraintDifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    ),
    scale=1,
)

DROID_IK_RELATIVE_ARM = TransformedOneShotDifferentialIKActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    body_name=["robotiq_base_link"],
    controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    # action_root_offset=TransformedOperationalSpaceControllerActionCfg.OffsetCfg(
    #     pos=read_metadata_from_usd_directory(EXPLICIT_UR5E_ROBOTIQ_2F85.spawn.usd_path).get("offset").get("pos"),
    #     rot=read_metadata_from_usd_directory(EXPLICIT_UR5E_ROBOTIQ_2F85.spawn.usd_path).get("offset").get("quat"),
    # ),
    scale=0.002,
)

ROBOTIQ_GRIPPER_BINARY_ACTIONS = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["finger_joint"],
    open_command_expr={"finger_joint": 0.0},
    close_command_expr={"finger_joint": 0.785398},
)

ROBOTIQ_COMPLIANT_JOINTS = DefaultJointPositionStaticActionCfg(
    asset_name="robot", joint_names=["left_inner_finger_joint", "right_inner_finger_joint"]
)

ROBOTIQ_MC_IK_ABSOLUTE = MultiConstraintsDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["joint.*"],
    body_name=["left_inner_finger", "right_inner_finger"],
    controller=MultiConstraintDifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    ),
    scale=1,
)


@configclass
class DROIDIkAbsoluteAction:
    arm = DROID_MC_IKABSOLUTE_ARM
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
    compliant_joints = ROBOTIQ_COMPLIANT_JOINTS


@configclass
class DROIDIkDeltaAction:
    arm = DROID_IK_RELATIVE_ARM
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
    compliant_joints = ROBOTIQ_COMPLIANT_JOINTS


@configclass
class DROIDJointPositionAction:
    arm = DROID_JOINT_POSITION
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
    compliant_joints = ROBOTIQ_COMPLIANT_JOINTS


@configclass
class DROIDRelativeJointPositionAction:
    arm = DROID_RELATIVE_JOINT_POSITION
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
    compliant_joints = ROBOTIQ_COMPLIANT_JOINTS


@configclass
class Robotiq2f85BinaryGripperAction:
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
    compliant_joints = ROBOTIQ_COMPLIANT_JOINTS
