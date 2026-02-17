class Env(ManagerBasedRLEnv):
    def reset(self, *args, **kwargs):
        self.sim.pause()
        # Call normal reset
        obs, info = super().reset(*args, **kwargs)
        self.sim.play()
        
        self.sim.step(render=True)
        return obs, info