from configs.environment_cfg import EnvCfg

from isaaclab.envs import DirectRLEnv


class Env(DirectRLEnv):
    '''
    The RL environment
    '''
    def __init__(
        self,
        cfg: EnvCfg,
        render_mode: str | None = None,
        **kwargs
    ) -> None:
        super().__init__(render_mode=render_mode,
                         **kwargs,)
        
        # Fill later
        self.idx_joints: list = [None]
        self.action_scale: float = self.cfg.action_scale