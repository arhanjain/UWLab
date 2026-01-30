from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm, ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg, EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from uwlab_tasks.manager_based.manipulation.reset_states.config.droid.rl_state_cfg import DROIDIkRelativeEvalCfg, BaseEventCfg
from uwlab_assets.robots.DROID import DROIDJointPositionAction
from isaaclab.utils import configclass, noise

from ... import mdp as task_mdp
import numpy as np

def gripper_pos(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[:, joint_indices]

    # rescale
    joint_pos = joint_pos / (np.pi / 4)

    return joint_pos


@configclass
class VisionCfg(ObsGroup):
    """Observations for policy group."""
    wrist_camera = ObsTerm(
        func=task_mdp.image, 
        params={"sensor_cfg": SceneEntityCfg("wrist_camera"), 
        "data_type": "rgb", "normalize": False}
        )
    external_camera = ObsTerm(
        func=task_mdp.image, 
        params={"sensor_cfg": SceneEntityCfg("external_camera"), 
        "data_type": "rgb", "normalize": False}
        )

    arm_joint_pos = ObsTerm(
        func=task_mdp.selected_joint_pos, 
        params={
            "asset_cfg": SceneEntityCfg("robot"), 
            "joint_names": [
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
            ]
        }
    )

    gripper_pos = ObsTerm(
        func=gripper_pos,
        noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1)
    )


    # end_effector_pose = ObsTerm(
    #     func=task_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
    #     params={
    #         "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
    #         "root_asset_cfg": SceneEntityCfg("robot"),
    #         "target_asset_offset_metadata_key": "gripper_offset",
    #         "root_asset_offset_metadata_key": "offset",
    #         "rotation_repr": "axis_angle",
    #     },
    # )
    def __post_init__(self):
        self.concatenate_terms = False
        self.enable_corruption = False


def make_insertive_object(usd_path: str):
    if "tri" in usd_path:
        usd_path = utils.retrieve_file_path_with_s3_support(usd_path)

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(10.0, 10.0, 10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_receptive_object(usd_path: str):
    if "tri" in usd_path:
        usd_path = utils.retrieve_file_path_with_s3_support(usd_path)

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(10.0, 10.0, 10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
variants = {
    "scene.insertive_object": {
        "cube": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/InsertiveCube/insertive_cube.usd"),
    },
    "scene.receptive_object": {
        "plate": make_receptive_object(f"./envs/assets/plate/plate.usd"),
    },
}


@configclass
class DROIDIkRelativeVisionEvalCfg(DROIDIkRelativeEvalCfg):
    def __post_init__(self):
        super().__post_init__()

        self.variants = variants

        # Add Light, and remove visibliliyt of table
        self.scene.env_spacing = 20.0
        # self.scene.light = AssetBaseCfg(
        #     prim_path="{ENV_REGEX_NS}/Light",
        #     spawn = sim_utils.SphereLightCfg(
        #         intensity=5000.0,
        #         color=(1.0, 1.0, 1.0),
        #     ),
        #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.42, -0.5, 1.0), rot=(0.0, 0.0, 0.0, 1.0)),
        # )
        self.scene.sky_light.spawn.intensity = 500.0

        # remove skylight
        # del self.scene.sky_light
        del self.scene.ground

        self.scene.table.spawn.visible = False
        self.scene.ur5_metal_support.spawn.visible = False
        # self.scene.ground.spawn.visible = False

        self.scene.wrist_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/robotiq_2f85_gripper/robotiq_base_link/wrist_camera",
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.8,
                focus_distance=28.0,
                horizontal_aperture=5.376,
                vertical_aperture=3.024,
                clipping_range=(0.001, 20.0),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.011, -0.031, -0.074),
                rot=(-0.420, 0.570, 0.576, -0.409),
                convention="opengl",
            ),
        )
        self.scene.splat = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/splat",
            spawn=sim_utils.UsdFileCfg(
                usd_path="./envs/assets/tri_droid_scene/combined.usd",
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.05, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
        )
        self.scene.external_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/external_camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos = (0.09827, -0.42967, 0.39172),
                rot = (0.85097, 0.44271, -0.13042, -0.25069),
                convention="opengl"
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                    focal_length=1.0476,
                    horizontal_aperture=2.5452,
                    vertical_aperture=1.4721,
            ),
            height=720,
            width=1280,
        )

        # Add Camera Observations
        self.observations.vision = VisionCfg()

        # Add Success Termination
        self.terminations.success = DoneTerm(
            func=task_mdp.success_reward_bool,
            time_out=True,
            )

        self.episode_length_s = 10.0
        self.rerender_on_reset = True
        
        # hack the scales
        self.scene.splat.spawn.scale = (10.0, 10.0, 10.0)
        self.scene.robot.spawn.scale = (10.0, 10.0, 10.0)
        self.scene.external_camera.offset.pos = [entry * 10.0 for entry in self.scene.external_camera.offset.pos]
        # self.scene.insertive_object.spawn.scale = (10.0, 10.0, 10.0)
        # self.scene.receptive_object.spawn.scale = (10.0, 10.0, 10.0)
        self.scene.table.spawn.scale = (10.0, 10.0, 10.0)
        self.scene.ur5_metal_support.spawn.scale = (10.0, 10.0, 10.0)
        # self.scene.light.init_state.pos = [entry * 10.0 for entry in self.scene.light.init_state.pos]
        # self.scene.light.spawn.intensity = 5000.0 * 30.0
        # del self.scene.splat


@configclass
class DROIDJointPosVisionEvalCfg(DROIDIkRelativeVisionEvalCfg):
    def __post_init__(self):
        super().__post_init__()

        self.actions = DROIDJointPositionAction() 