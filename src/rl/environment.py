import yaml
import torch
import numpy as np
from collections.abc import Sequence
from typing import Any

from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg

from configs.python.scene_cfg import SceneCfg
from configs.python.environment_cfg import EnvCfg

    
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
        # Extract panda from scene
        self.panda = self.scene['panda']
        # Extract panda joint IDs
        arm_joint_names = ['panda_link.*']
        self.arm_joint_ids: np.ndarray = self.panda.find_bodies(arm_joint_names)[0]
    
    
    def _step_impl(
        self,
        actions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        obs, rewards, terminated, truncated, info = super()._step_impl(actions)
        # Perform interval-based domain randomization
        self.event_manager.step(self.physics_dt)
        # Update panda buffers
        self.panda.update(self.physics_dt)
        return obs, rewards, terminated, truncated, info
    
    
    def _pre_physics_step(
        self, 
        actions: torch.Tensor,
    ) -> None:
        self.actions = self.cfg.action_scale * actions.clone()
        
    
    def _apply_action(self,) -> None:
        self.panda.set_joint_effort_target(
                self.actions, 
                joint_ids=self.arm_joint_ids,
            )
    
    
    def _get_observations(self,) -> dict[str, torch.Tensor]:
        observations: torch.Tensor = torch.cat(
            (
                self.panda.data.joint_pos[:, self.arm_joint_ids].unsqueeze(dim=1,),
                self.panda.data.joint_vel[:, self.arm_joint_ids].unsqueeze(dim=1,),
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
            self.panda,
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
            env_ids = self.panda._ALL_INDICES
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