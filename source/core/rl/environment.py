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

from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from isaaclab.sim.utils.stage import get_current_stage
from pxr import UsdGeom

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
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        # Extract assets of interest from scene
        self.robot = self.scene['robot']
        self.contact_forces = self.scene['contact_forces']
        for i in range(config.config['scene']['object']['n_assets']):
            setattr(self, f'object_{i}', self.scene[f'object_{i}'])
        #self._define_markers()
        
        self.hand_link_idx = self.robot.find_bodies('panda_link7')[0][0]
        self.lf_link_idx = self.robot.find_bodies('panda_leftfinger')[0][0]
        self.rf_link_idx = self.robot.find_bodies('panda_rightfinger')[0][0]
        self.arm_joint_ids = self.robot.find_bodies(['panda_link.*'])[0]
        # Center of robot's joint ranges
        self.joint_centers: torch.Tensor = torch.mean(self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids, :], dim=-1)
        
        
        stage = get_current_stage()
        hand_pose = self.get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
        )
        lf_pose = self.get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
        )
        rf_pose = self.get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
        )
        
        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lf_pose[0:3] + rf_pose[0:3]) / 2.0
        finger_pose[3:7] = lf_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])
        robot_local_ee_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3],
        )
        
        robot_local_pose_pos += torch.tensor([0.0, 0.04, 0.0], device=self.sim.device)
        self.robot_local_ee_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_ee_rot = robot_local_ee_pose_rot.repeat((self.num_envs, 1))
        
        self.robot_ee_pos: torch.Tensor = torch.zeros((self.num_envs, 3,), device=self.sim.device)
        self.robot_ee_rot: torch.Tensor = torch.zeros((self.num_envs, 4,), device=self.sim.device)
        self.object_pos: torch.Tensor = torch.zeros((self.num_envs, config.config['scene']['object']['n_assets'], 3,), device=self.sim.device)
        self.object_rot: torch.Tensor = torch.zeros((self.num_envs, config.config['scene']['object']['n_assets'], 4,), device=self.sim.device)
        
        self.robot_up_axis: torch.Tensor = torch.tensor([0.0, 1.0, 0.0], device=self.sim.device).repeat((self.num_envs, 1,))
        self.robot_for_axis: torch.Tensor = torch.tensor([0.0, 0.0, 1.0], device=self.sim.device).repeat((self.num_envs, 1,))
        self.object_up_axis: torch.Tensor = torch.tensor([0.0, 0.0, 1.0], device=self.sim.device).repeat((self.num_envs, 1,))
        self.object_in_axis: torch.Tensor = torch.tensor([-1.0, 0.0, 0.0], device=self.sim.device).repeat((self.num_envs, 1,))
        
        # Define the OSC for taskspace action handling
        self.osc = get_osc(self.num_envs, self.sim.device,)
        # Reward reporting
        self._episode_rewards = {
            key: torch.zeros((self.num_envs), device=self.sim.device)
            for key in ["dist_term", "rot_term", "binary_contact_term", "contact_term", "finger_dist_term", "object_height_term", "sparse_height_term", "velocity_term", "passive_term"]
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
        self.actions = torch.cat((
            actions, # OSC controls + EE joint position
            actions[..., 19].unsqueeze(-1) # Copied EE joint position
        ), dim=1,)
        self.actions *= self.cfg.action_scale
        
        _, _, _, ee_pose_b, _, _, _, _ = update_states(
            robot=self.robot,
            sim_dt=self.sim.get_physics_dt(),
            num_envs=self.num_envs,
            ee_frame_idx=self.lf_link_idx,
            arm_joint_ids=self.arm_joint_ids,
            contact_forces=self.contact_forces,
        )
        # Updates the command and specialized position/quaternion orientation target
        command, ee_target_pose_b = update_target(
                self.osc,
                self.num_envs,
                self.sim.device,
                self.actions[..., :19],
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
        '''marker_pos: torch.Tensor = self.ee_pos
        marker_pos[:, 2] += config.config["scene"]["marker"]["translation"]'''
        # Compute marker orientation quat
        '''target_vect_orient: torch.Tensor = F.normalize(
            (self.ee_pos),
            dim=1
        )'''
        #target_quat: torch.Tensor = vect_to_quat(target_vect_orient, self.local_forward)
        
        '''ee_tool_axis = torch.tensor([0.0, 0.0, 1.0], device=self.sim.device).repeat((self.num_envs, 1))
        ee_tool_vect = quat_apply(self.ee_quat, ee_tool_axis)
        #ee_tool_quat = vect_to_quat(ee_tool_vect, ee_tool_axis)'''
        
        # Update markers
        '''self.markers.visualize(
            translations=marker_pos.repeat((2, 1),)[0:self.num_envs, :],
            orientations=torch.concat(
                (ee_tool_quat,),
                dim=0,
            ),
        )'''
    
    
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
            ee_frame_idx=self.lf_link_idx,
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
        
        
    def _compute_intermediate_values(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """
        Computes values needed for rewards and marker visualization
        
        Args:
            env_ids (torch.Tensor): Environment ids
        """
        # Resolve ids
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
            
        hand_pos = self.robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self.robot.data.body_quat_w[env_ids, self.hand_link_idx]
        
        self.robot_ee_rot[env_ids], self.robot_ee_pos[env_ids] = tf_combine(
            hand_rot, hand_pos, self.robot_local_ee_rot[env_ids], self.robot_local_ee_pos[env_ids],
        )
        self.object_pos = torch.stack(
            [getattr(self, f'object_{i}').data.body_pos_w[env_ids] for i in range(config.config['scene']['object']['n_assets'])],
            dim=1,
        ).squeeze(-2)
        self.object_rot = torch.stack(
            [getattr(self, f'object_{i}').data.body_quat_w[env_ids] for i in range(config.config['scene']['object']['n_assets'])],
            dim=1,
        ).squeeze(-2)
        
        #self.ee_orient: torch.Tensor = F.normalize(quat_apply(self.ee_quat, self.local_forward,), dim=1,) # Quaternion applied to a unit vector should result in a unit vector, but floating-point errors can accumulate
        #self.targ_orient: torch.Tensor = F.normalize((self.cube_pos - self.ee_pos), dim=1,)
        
        self.object_poses: torch.Tensor = torch.stack([getattr(self, f'object_{i}').data.root_link_pose_w for i in range(config.config['scene']['object']['n_assets'])], dim=1)
    
    
    def _get_observations(self,) -> dict[str, torch.Tensor]:
        # Get joint localization
        joint_pos: torch.Tensor = self.robot.data.joint_pos[..., :-1]
        joint_vel: torch.Tensor = self.robot.data.joint_vel[..., :-1]
        observations: torch.Tensor = torch.cat((
                joint_pos, joint_vel,
            ),
            dim=-1,
        )
        return observations
    
    
    def _get_rewards(self,) -> torch.Tensor:
        """
        Gets the rewards for all environments

        Returns:
            torch.Tensor: Rewards
        """
        # Get episode length
        episode_length: torch.Tensor = self.episode_length_buf
        env_ids = torch.arange(self.num_envs, device=self.device)
        # Get focal object
        distances: torch.Tensor = torch.norm(self.robot_ee_pos.unsqueeze(1).repeat(1, config.config['scene']['object']['n_assets'], 1) - self.object_pos, dim=2)
        dist_terms: torch.Tensor = 1.0 / (1.0 + distances**2)
        dist_terms = torch.where(distances <= 0.02, dist_terms * 2, dist_terms)
        # Base rewards off the closest object
        dist_term, focal_indicies = torch.max(dist_terms, dim=1,)
        
        # Encourage ee alignment to object
        axis1 = tf_vector(self.robot_ee_rot[env_ids], self.robot_for_axis)
        axis2 = tf_vector(self.object_rot[env_ids, focal_indicies], self.object_in_axis)
        axis3 = tf_vector(self.robot_ee_rot[env_ids], self.robot_up_axis)
        axis4 = tf_vector(self.object_rot[env_ids, focal_indicies], self.object_up_axis)
        
        dot1 = torch.bmm(axis1.view(self.num_envs, 1, 3), axis2.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        dot2 = torch.bmm(axis3.view(self.num_envs, 1, 3), axis4.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        
        rot_term: torch.Tensor = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)
        
        robot_lf_pos: torch.Tensor = self.robot.data.body_pos_w[:, self.lf_link_idx]
        robot_rf_pos: torch.Tensor = self.robot.data.body_pos_w[:, self.rf_link_idx]
        
        lf_dist: torch.Tensor = robot_lf_pos[:, 2] - self.object_pos[env_ids, focal_indicies, 2]
        rf_dist: torch.Tensor = robot_rf_pos[:, 2] - self.object_pos[env_ids, focal_indicies, 2]
        finger_dist_term: torch.Tensor = torch.zeros_like(lf_dist)
        finger_dist_term += torch.where(lf_dist < 0, lf_dist, torch.zeros_like(lf_dist))
        finger_dist_term += torch.where(rf_dist < 0, rf_dist, torch.zeros_like(rf_dist))
        
        # Presumably, the object that is being contacted is also the closest for simplicity
        forces: torch.Tensor = torch.norm(self.contact_forces.data.net_forces_w.squeeze(1), dim=1)
        scaled_forces: torch.Tensor = forces / 100.0 # Scale forces for numerical stability
        is_contacting: torch.Tensor = scaled_forces > 1.0 # Counter noise
        
        # Find the average prismatic position
        ee_jointspace_pos: torch.Tensor = torch.mean(self.robot.data.joint_pos[:, 7:8], dim=1,)
        # Get object height
        object_height: torch.Tensor = self.object_poses[env_ids, focal_indicies, 2]
        is_above_height: torch.Tensor = object_height >= config.config['env']['reward']['sparse_height_threshold']
        
        mean_velocity: torch.Tensor = torch.mean(torch.abs(self.robot.data.joint_vel), dim=1,)
        rewards = {
            "dist_term":              config.config["env"]["reward"]["dist_coef"] * dist_term, # Reduce distance from object
            "rot_term":               config.config['env']['reward']['rot_coef'] * rot_term,
            "binary_contact_term":    config.config["env"]["reward"]["binary_contact_coef"] * is_contacting, # 1/0 Make contact with object
            "contact_term":           config.config["env"]["reward"]["contact_coef"] * -scaled_forces, # Discourage forceful contacts
            "finger_dist_term":       config.config['env']['reward']['finger_dist_coef'] * finger_dist_term,
            "object_height_term":     config.config["env"]["reward"]["object_height_coef"] * object_height, # Lift object
            "sparse_height_term":     config.config["env"]["reward"]["sparse_height_coef"] * is_above_height, # Lift object past boundary ("success condition")
            "velocity_term":          config.config["env"]["reward"]["velocity_coef"] * -mean_velocity, # Punish large actions
            #"passive_term":           config.config["env"]["reward"]["passive_coef"] * -episode_length, # Reduce episode length
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
        
    
    def get_env_local_pose(
        self,
        env_pos: torch.Tensor, 
        xformable: UsdGeom.Xformable,
    ) -> torch.Tensor:
        '''
        Compute pose in env-local coordinates
        
        Args:
            env_pos (torch.Tensor): Environment origin
            xformable (torch.Tensor): Object
        
        Returns:
            torch.Tensor: Env frame pose (pos, quat)
        '''
        # Compute world-frame poses
        world_transform = xformable.ComputeLocalToWorldTransform(0)
        world_pos = world_transform.ExtractTranslation()
        world_quat = world_transform.ExtractRotationQuat()
        # Compute env-frame poses
        pos: torch.Tensor = torch.tensor([
            world_pos[0] - env_pos[0], world_pos[1] - env_pos[1], world_pos[2] - env_pos[2],
        ], device=self.sim.device)
        quat: torch.Tensor = torch.tensor([
            world_quat.real, world_quat.imaginary[0], world_quat.imaginary[1], world_quat.imaginary[2],
        ], device=self.sim.device)
        return torch.cat((
                pos, quat,
            ),
            dim=0,
        )