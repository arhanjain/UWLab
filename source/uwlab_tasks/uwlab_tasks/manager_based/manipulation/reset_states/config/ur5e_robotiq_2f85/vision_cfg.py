from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm, ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from uwlab_tasks.manager_based.manipulation.reset_states.config.ur5e_robotiq_2f85.rl_state_cfg import Ur5eRobotiq2f85RelCartesianOSCEvalCfg
from ... import mdp as task_mdp

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
    joint_pos = ObsTerm(
        func=task_mdp.selected_joint_pos, 
        params={
            "asset_cfg": SceneEntityCfg("robot"), 
            "joint_names": [
                "shoulder_pan_joint", 
                "shoulder_lift_joint", 
                "elbow_joint", 
                "wrist_1_joint", 
                "wrist_2_joint", 
                "wrist_3_joint",
                "finger_joint",
            ]
        }
    )


    end_effector_pose = ObsTerm(
        func=task_mdp.target_asset_pose_in_root_asset_frame_with_metadata,
        params={
            "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "root_asset_cfg": SceneEntityCfg("robot"),
            "target_asset_offset_metadata_key": "gripper_offset",
            "root_asset_offset_metadata_key": "offset",
            "rotation_repr": "axis_angle",
        },
    )
    def __post_init__(self):
        self.concatenate_terms = False
        self.enable_corruption = False

@configclass
class Ur5eRobotiq2f85RelCartesianOSCVisionEvalCfg(Ur5eRobotiq2f85RelCartesianOSCEvalCfg):
    def __post_init__(self):
        super().__post_init__()

        # Add Curtains & Light
        self.scene.env_spacing = 2.5
        self.scene.light = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Light",
            spawn = sim_utils.SphereLightCfg(
                intensity=30000.0,
                color=(1.0, 1.0, 1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.42, -0.5, 1.0), rot=(0.0, 0.0, 0.0, 1.0)),
        )

        self.scene.curtain_left = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/CurtainLeft",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.4, -0.68, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
            spawn=sim_utils.CuboidCfg(
                size=(0.01, 1.0, 1.125),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 0.0)
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=False,
                )
            ),
            )

        self.scene.curtain_back = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/CurtainBack",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.15, 0.0, 0.519), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CuboidCfg(
                size=(0.01, 1.3, 1.125),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 0.0)
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=False,
                )
            ),
        )

        self.scene.curtain_right = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/CurtainRight",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.4, 0.68, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
            spawn=sim_utils.CuboidCfg(
                size=(0.01, 1.0, 1.125),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 0.0)
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=False,
                )
            ),
        )

        # Add Cameras
        self.scene.wrist_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/wrist_camera",
            offset=TiledCameraCfg.OffsetCfg(pos=(-0.02, 0.0, -0.08), rot=(0.40558, -0.57923, -0.57923, 0.40558), convention="opengl"),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
            ),
            width=640,
            height=480,
        )
        self.scene.external_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/external_camera",
            offset=TiledCameraCfg.OffsetCfg(pos=(1.7, 0.0, 0.6), rot=(0.57923, 0.40558, 0.40558, 0.57923), convention="opengl"),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
            ),
            width=640,
            height=480,
        )

        # Add Camera Observations
        self.observations.vision = VisionCfg()

        # Add Success Termination
        self.terminations.success = DoneTerm(
            func=task_mdp.success_reward_bool,
            time_out=True,
            )

        self.episode_length_s = 10.0
