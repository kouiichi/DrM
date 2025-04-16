from gymnasium.envs.registration import register

register(
    id='GymSpaceRobotEnv-v0',
    entry_point="SpaceRobotEnv.envs:GymSpaceRobotEnv",
    max_episode_steps=64,
)