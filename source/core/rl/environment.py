import torch
import torch.nn.functional as F
import yaml
import numpy as np
from collections.abc import Sequence
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from isaaclab.assets import Articulation

from isaaclab.sensors import Camera, TiledCamera, TiledCameraCfg, ContactSensor
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from source.configs.python.scene_cfg import SceneCfg
from source.configs.python.environment_cfg import EnvCfg
from source.sim.osc import get_osc, update_states, update_target, convert_to_task_frame
from source.utils.math import vect_to_quat
from source.utils.timer import timer
import source.utils.config as config

import omni.physx as _physx

    
class Env(DirectRLEnv):
    '''
    RL environment
    '''
    def __init__(
        self,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(EnvCfg(), render_mode, **kwargs)
        # Extract assets of interest from scene
        self.robot = self.scene['robot']
        self.contact_forces = self.scene['contact_forces']
        self._define_markers()
        # Extract robot joint IDs
        ee_frame_name = 'panda_leftfinger'
        arm_joint_names = ['panda_link.*']
        self.ee_frame_idx: int = self.robot.find_bodies(ee_frame_name)[0][0]
        self.arm_joint_ids: np.ndarray = self.robot.find_bodies(arm_joint_names)[0]
        # Local forward for computing orientation quats
        self.local_forward: torch.Tensor = torch.tensor([1.0, 0.0, 0.0], device=self.sim.device,).repeat((self.num_envs, 1),)
        # Center of robot's joint ranges
        self.joint_centers: torch.Tensor = torch.mean(self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids, :], dim=-1)
        
        # Define the OSC for taskspace action handling
        self.osc = get_osc(self.num_envs, self.sim.device,)
        # Reward reporting
        self._episode_rewards = {
            key: torch.zeros((self.num_envs), device=self.sim.device)
            for key in ["dist_term", "binary_contact_term", "contact_term", "ee_open_term", "object_height_term", "sparse_height_term", "velocity_term", "passive_term"]
        }
        
    
    def _define_markers(self,) -> None:
        """
        Create orientation markers
        """
        markers_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
            prim_path="/Visuals/goal_marker",
            markers={
                "ee_orientation": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=tuple(config.config["scene"]["marker"]["scale"]),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
                #"targ_orientation": sim_utils.UsdFileCfg(
                #    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                #    scale=tuple(config["scene"]["marker"]["scale"]),
                #    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                #),
            },
        )
        self.markers = VisualizationMarkers(markers_cfg,)
    
    
    def _pre_physics_step(
        self,
        actions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        self.actions = actions
        
        _, _, _, ee_pose_b, _, _, _, _ = update_states(
            robot=self.robot,
            sim_dt=self.sim.get_physics_dt(),
            num_envs=self.num_envs,
            ee_frame_idx=self.ee_frame_idx,
            arm_joint_ids=self.arm_joint_ids,
            contact_forces=self.contact_forces,
        )
        
        # Updates the command and specialized position/quaternion orientation target
        command, ee_target_pose_b = update_target(
                self.osc,
                self.num_envs,
                self.sim.device,
                actions,
            )
        # Set the OSC command
        command, task_frame_pose_b = convert_to_task_frame(
            self.osc,
            command,
            ee_target_pose_b,
        )
        self.osc.set_command(
            command=command,
            current_ee_pose_b=ee_pose_b,
            current_task_frame_pose_b=task_frame_pose_b,
        )
        
        self._compute_intermediate_values()
        
        # Place markers above ee
        marker_pos: torch.Tensor = self.ee_pos
        marker_pos[:, 2] += config.config["scene"]["marker"]["translation"]
        # Compute marker orientation quat
        target_vect_orient: torch.Tensor = F.normalize(
            (self.ee_pos),
            dim=1
        )
        target_quat: torch.Tensor = vect_to_quat(target_vect_orient, self.local_forward)
        
        ee_tool_axis = torch.tensor([0.0, 0.0, 1.0], device=self.sim.device).repeat((self.num_envs, 1))
        ee_tool_vect = quat_apply(self.ee_quat, ee_tool_axis)
        ee_tool_quat = vect_to_quat(ee_tool_vect, ee_tool_axis)
        
        # Update markers
        self.markers.visualize(
            translations=marker_pos.repeat((2, 1),)[0:self.num_envs, :],
            orientations=torch.concat(
                (ee_tool_quat,),
                dim=0,
            ),
        )
    
    
    def _apply_action(self,) -> None:
        (
            jacobian_b,
            mass_mat,
            gravity,
            ee_pose_b, 
            ee_vel_b,
            ee_force_b,
            joint_pos,
            joint_vel
        ) = update_states(
            robot=self.robot,
            sim_dt=self.sim.get_physics_dt(),
            num_envs=self.num_envs,
            ee_frame_idx=self.ee_frame_idx,
            arm_joint_ids=self.arm_joint_ids,
            contact_forces=self.contact_forces,
        )
        # Get joint-level commands
        joint_efforts: torch.Tensor = self.osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            current_ee_force_b=ee_force_b,
            mass_matrix=mass_mat,
            gravity=gravity,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
            nullspace_joint_pos_target=self.joint_centers,
        )
        # Write joint effort
        self.robot.set_joint_effort_target(
                joint_efforts, 
                joint_ids=self.arm_joint_ids,
            )
        self.robot.update(self.physics_dt)
        self.robot.write_data_to_sim()
        
        
    def _compute_intermediate_values(self,) -> None:
        """
        Computes values needed for rewards and marker visualization
        """
        # Compute locations in world frame, then distance
        self.ee_pos: torch.Tensor = self.robot.data.body_pos_w[:, self.ee_frame_idx]
        # Compute quats
        self.ee_quat: torch.Tensor = self.robot.data.body_quat_w[:, self.ee_frame_idx]
        
        self.ee_orient: torch.Tensor = F.normalize(quat_apply(self.ee_quat, self.local_forward,), dim=1,) # Quaternion applied to a unit vector should result in a unit vector, but floating-point errors can accumulate
        #self.targ_orient: torch.Tensor = F.normalize((self.cube_pos - self.ee_pos), dim=1,)
        
        for i in range(config.config['scene']['object']['n_assets']):
            # Get positions
            self.object_positions: list[torch.Tensor] = [getattr(self, f'object_{i}').data.root_link_pose_w[0:3]]
    
    
    def _get_observations(self,) -> dict[str, torch.Tensor]:
        # Get joint localization
        joint_pos: torch.Tensor = self.robot.data.joint_pos[..., :-1]
        joint_vel: torch.Tensor = self.robot.data.joint_vel[..., :-1]
        observations: torch.Tensor = torch.cat((
                joint_pos, joint_vel,
            ),
            dim=-1,
        )
        obs_dict: dict[str, torch.Tensor] = {'policy': observations}
        return obs_dict
    
    
    def _get_rewards(self,) -> torch.Tensor:
        """
        Gets the rewards for all environments

        Returns:
            torch.Tensor: Rewards
        """
        # Get episode length
        episode_length: torch.Tensor = self.episode_length_buf
        # Get focal object
        distances: torch.Tensor = torch.stack([torch.cdist(self.ee_pos, obj_pos) for obj_pos in self.object_position], dim=0,)
        # Base rewards off the closest object
        focal_indicies: torch.Tensor = torch.argmax(distances, dim=0,)
        
        distance: torch.Tensor = distances[focal_indicies, ...]
        # Presumably, the object that is being contacted is also the closest for simplicity
        is_contacting: torch.Tensor = self.contact_forces.data.net_forces_w > 1.0 # Counter noise
        scaled_forces: torch.Tensor = self.contact_forces.data.net_forces_w / 100.0 # Scale forces for numerical stability
        # Find the average prismatic position
        ee_jointspace_pos: torch.Tensor = torch.mean(self.robot.data.joint_pos[:, 7:8], dim=1,)
        object_height: torch.Tensor = self.object_positions[focal_indicies, 2]
        is_above_height: torch.Tensor = object_height >= config.config['env']['reward']['sparse_height_threshold']
        
        mean_velocity: torch.Tensor = torch.mean(self.robot.data.joint_vel, dim=1,)
        
        rewards = {
            "dist_term":           config.config["env"]["reward"]["dist_coef"] * -distance, # Reduce distance from object
            "binary_contact_term": config.config["env"]["reward"]["binary_contact_coef"] * is_contacting, # 1/0 Make contact with object
            "contact_term":        config.config["env"]["reward"]["contact_coef"] * -scaled_forces, # Discourage forceful contacts
            "ee_open_term":        config.config["env"]["reward"]["ee_open_coef"] * ee_jointspace_pos, # Keep the end-effector open
            "object_height_term":  config.config["env"]["reward"]["object_height_coef"] * object_height, # Lift object
            "sparse_height_term":  config.config["env"]["reward"]["sparse_height_coef"] * is_above_height, # Lift object past boundary ("success condition")
            "velocity_term":       config.config["env"]["reward"]["velocity_coef"] * -mean_velocity, # Punish large actions
            "passive_term":        config.config["env"]["reward"]["passive_coef"] * -episode_length, # Reduce episode length
        }
        # Save rewards for logging
        for key, reward in rewards.items():
            self._episode_rewards[key] += reward
        # Cumulative reward
        return torch.sum(
            torch.stack(tuple(rewards.values()), dim=1,),
            dim=1,
        )
        
        
    def _get_dones(self,) -> tuple:
        """
        Gets the boolean episode completion flags for all environments
        
        Returns:
            tuple[bool, bool]: A tuple containing the (terminated, truncated) completion flags
        """
        # Truncation term
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Return both
        return torch.full_like(torch.empty(self.num_envs), False, device=self.device), time_out # Termination is excluded
    
    
    @timer
    def _reset_idx(
        self,
        env_ids: Sequence[int] | None = None
    ) -> None:
        # Default is all environments
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # Pause sim for USD modifications
        self.sim.pause()
        super()._reset_idx(env_ids)
        self.sim.play()
        # Reset OSC and sim
        self.sim.reset()
        self.osc.reset()