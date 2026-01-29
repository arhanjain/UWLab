from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    JointPositionActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.utils import configclass

from uwlab.controllers.differential_ik_cfg import MultiConstraintDifferentialIKControllerCfg
from uwlab.envs.mdp.actions.actions_cfg import (
    DefaultJointPositionStaticActionCfg,
    MultiConstraintsDifferentialInverseKinematicsActionCfg,
)

import torch

### ActionCfg ###
class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    # override
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: close, false: open
            binary_mask = actions > 0.5
        # compute the command
        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type = BinaryJointPositionZeroToOneAction

"""
UR5E ROBOTIQ 2F85 ACTIONS
"""
FRANKA_JOINT_POSITION: JointPositionActionCfg = JointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    scale=1.0,
    use_default_offset=False,
)


FRANKA_MC_IKABSOLUTE_ARM = MultiConstraintsDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    body_name=["base_link"],
    controller=MultiConstraintDifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    ),
    scale=1,
)

FRANKA_MC_IKDELTA_ARM = MultiConstraintsDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    body_name=["base_link"],
    controller=MultiConstraintDifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    scale=0.5,
)

ROBOTIQ_GRIPPER_BINARY_ACTIONS = BinaryJointPositionZeroToOneActionCfg(
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
class DROIDJointPositionAction:
    jointpos = FRANKA_JOINT_POSITION
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
    compliant_joints = ROBOTIQ_COMPLIANT_JOINTS

@configclass
class DROIDMcIkAbsoluteAction:
    arm = FRANKA_MC_IKABSOLUTE_ARM
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS

@configclass
class DROIDBinaryGripperAction:
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
    compliant_joints = ROBOTIQ_COMPLIANT_JOINTS

# @configclass
# class Ur5eRobotiq2f85IkAbsoluteAction:
#     arm = UR5E_MC_IKABSOLUTE_ARM
#     gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
#     compliant_joints = ROBOTIQ_COMPLIANT_JOINTS


# @configclass
# class Ur5eRobotiq2f85McIkDeltaAction:
#     arm = UR5E_MC_IKDELTA_ARM
#     gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
#     compliant_joints = ROBOTIQ_COMPLIANT_JOINTS


# @configclass
# class Ur5eRobotiq2f85JointPositionAction:
#     arm = UR5E_JOINT_POSITION
#     gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
#     compliant_joints = ROBOTIQ_COMPLIANT_JOINTS


# @configclass
# class Ur5eRobotiq2f85RelativeJointPositionAction:
#     arm = UR5E_RELATIVE_JOINT_POSITION
#     gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
#     compliant_joints = ROBOTIQ_COMPLIANT_JOINTS


# @configclass
# class Robotiq2f85BinaryGripperAction:
#     gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
#     compliant_joints = ROBOTIQ_COMPLIANT_JOINTS
