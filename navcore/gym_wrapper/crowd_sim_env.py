import gymnasium as gym

from navcore.builder.environment_builder import EnvironmentBuilder


class CrowdSimEnv(gym.Env):
    def __init__(self, env_builder: EnvironmentBuilder):
        super().__init__()

        self.env_builder = env_builder
        self.env = self.env_builder.build_environment()

        self.action_space = self.env.robot.action_space
        self.observation_space = self.env.robot.observation_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.environment = self.env_builder.reset(random_seed=seed)
        observation = self._get_observation()
        info = {}
        return observation, info
