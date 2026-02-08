import yaml
import torch
import numpy as np
from collections.abc import Sequence
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, TiledCamera, ContactSensor

from src.configs.python.scene_cfg import SceneCfg
from src.configs.python.environment_cfg import EnvCfg
from src.sim.osc import get_osc, update_states, update_target, convert_to_task_frame

    
class Env(DirectRLEnv):
    '''
    RL environment
    '''
    def __init__(
        self,
        env_cfg: EnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(env_cfg, render_mode, **kwargs)
        # Extract assets of interest from scene
        self.robot = self.scene['robot']
        self.contact_forces = self.scene['contact_forces']
        
        # Extract robot joint IDs
        ee_frame_name = 'panda_leftfinger'
        arm_joint_names = ['panda_link.*']
        self.ee_frame_idx: int = self.robot.find_bodies(ee_frame_name)[0][0]
        self.arm_joint_ids: np.ndarray = self.robot.find_bodies(arm_joint_names)[0]
        # Center of robot's joint ranges
        self.joint_centers: torch.Tensor = torch.mean(self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids, :], dim=-1)
        
        # Define the OSC for taskspace action handling
        self.osc = get_osc(
            self.num_envs, self.sim.device,
        )
        
        
    def _setup_scene(self,):
        # Set up sensors
        self._contact_forces = ContactSensor(self.cfg.contact_forces)
        self.scene.sensors["contact_forces"] = self._contact_forces
        self._camera = Camera(self.cfg.camera)
        self.scene.sensors["camera"] = self._camera
        super()._setup_scene()
    
    
    def _step_impl(
        self,
        actions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        obs, rewards, terminated, truncated, info = super()._step_impl(actions)
        # Perform interval-based domain randomization
        self.event_manager.step(self.physics_dt)
        
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
        self.osc.reset()
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
        
        # Update robot buffers
        self.robot.update(self.physics_dt)
        return obs, rewards, terminated, truncated, info
    
    
    def _pre_physics_step(
        self, 
        actions: torch.Tensor,
    ) -> None:
        self.actions = self.cfg.action_scale * actions.clone()
        
    
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
        self.robot.set_joint_effort_target(
                joint_efforts, 
                joint_ids=self.arm_joint_ids,
            )
    
    
    def _get_observations(self,) -> dict[str, torch.Tensor]:
        observations: torch.Tensor = torch.cat(
            (
                self.robot.data.joint_pos[:, self.arm_joint_ids].unsqueeze(dim=1,),
                self.robot.data.joint_vel[:, self.arm_joint_ids].unsqueeze(dim=1,),
            ),
            dim=-1,
        )
        obs_dict: dict[str, torch.Tensor] = {'policy': observations}
        return obs_dict
    
    
    def _get_rewards(self,) -> torch.Tensor:
        '''reward: torch.Tensor = Env.compute_rewards(
            self.cfg.rew_scale_grasp,
            self.cfg.rew_scale_duration,
            self.cfg.rew_scale_distance,
            self.cfg.rew_scale_drop,
            self.cfg.rew_scale_contact,
            self.robot,
            self.reset_terminated
        )'''
        reward: torch.Tensor = torch.zeros((self.num_envs))
        return reward
    
    
    def _get_dones(self,) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, False # For debugging, environment will reset if not due to truncation
    
    
    def _reset_idx(
        self,
        env_ids: Sequence[int] | None = None
    ) -> None:
        # Pause sim for USD modifications
        self.sim.pause()
        # Default is all environments
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # Randomize domains set to mode "reset"
        self.event_manager.reset(env_ids)
        self.sim.play()
        
    
    @classmethod
    @torch.jit.script
    def compute_rewards(
        cls,
    ) -> float:
        return 0.0