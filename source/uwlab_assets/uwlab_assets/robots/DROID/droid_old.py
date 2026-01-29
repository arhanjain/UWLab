
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ROBOTIQ_2F85_DEFAULT_JOINT_POS = {
    "finger_joint": 0.0,
    "right_outer.*": 0.0,
    "left_outer.*": 0.0,
    "left_inner_finger_knuckle_joint": 0.0,
    "right_inner_finger_knuckle_joint": 0.0,
    "left_inner_finger_joint": -0.785398,
    "right_inner_finger_joint": 0.785398,
}

NVIDIA_DROID = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str("./envs/assets/patrick_droid/franka_robotiq_gripper.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(1, 0, 0, 0),
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -1 / 5 * np.pi,
            "panda_joint3": 0.0,
            "panda_joint4": -4 / 5 * np.pi,
            "panda_joint5": 0.0,
            "panda_joint6": 3 / 5 * np.pi,
            "panda_joint7": 0,
            # "finger_joint": 0.0,
            # "right_outer.*": 0.0,
            # "left_inner.*": 0.0,
            # "right_inner.*": 0.0,
            **ROBOTIQ_2F85_DEFAULT_JOINT_POS,
        },
    ),
    soft_joint_pos_limit_factor=1,
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=400.0,
            damping=80.0,
        ),
        # "gripper": ImplicitActuatorCfg(
        #     joint_names_expr=["finger_joint"],
        #     stiffness=None,
        #     damping=None,
        #     effort_limit=200.0,
        #     velocity_limit=5.0,  # 2.175,
        # ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            stiffness=17,
            damping=5,
            effort_limit_sim=165,
        ),
        "inner_finger": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            stiffness=0.2,
            damping=0.02,
            effort_limit_sim=0.5,
        ),
    },
)

ROBOTIQ = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/RobotiqGripper",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str("./envs/assets/patrick_gripper/robotiq_2f85_gripper.usd"),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=36, solver_velocity_iteration_count=0
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0.1), rot=(1, 0, 0, 0), joint_pos=ROBOTIQ_2F85_DEFAULT_JOINT_POS
    ),
    actuators={
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            stiffness=17,
            damping=5,
            effort_limit_sim=165,
        ),
        "inner_finger": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            stiffness=0.2,
            damping=0.02,
            effort_limit_sim=0.5,
        ),
    },
    soft_joint_pos_limit_factor=1,
)
