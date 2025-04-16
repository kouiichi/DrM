import os
import time

import cv2
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image

import dm_control
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.utils import containers
from dm_control import viewer

from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

# 定义环境版本
_DEFAULT_TIME_LIMIT = 64
_CONTROL_TIMESTEP = 1
_MAX_STEPS = int(_DEFAULT_TIME_LIMIT / _CONTROL_TIMESTEP)

current_dir = os.path.dirname(os.path.abspath(__file__))
_XML_PATH = os.path.join(current_dir, 'SpaceRobotEnv', 'assets', 'spacerobot', 'spacerobot.xml')

if not os.path.exists(_XML_PATH):
    raise FileNotFoundError(f"XML file not found at {current_dir}. Please check the path.")


class SpaceRobot(base.Task):
    """A SpaceRobot task."""

    def __init__(self, physics, random=None):
        # 初始化父类
        super(SpaceRobot, self).__init__(random=random)
        
        # 加载模型
        self._physics = physics

        # 定义关键实体ID
        self._chasersat_body_id = self._physics.model.body('chasersat').id
        self._targetsat_body_id = self._physics.model.body('targetsat').id
        self._gripper_body_id = self._physics.model.body('gripper_base').id
        self._handle_body_id = self._physics.model.body('handle').id
        self._camera_id = self._physics.model.camera('camera').id
        self._camera_body_id = self._physics.model.body('camera_body').id

        self.chasersat_mass = self._physics.model.body_mass[self._chasersat_body_id]
        self.n = 0.00113

        self.init_dist = np.linalg.norm(physics.data.body(self._gripper_body_id).xpos.copy() - physics.data.body(self._handle_body_id).xpos.copy())

    def initialize_episode(self, physics):
        """ Initialize the environment at the start of each episode. """

        # 重置物理仿真
        physics.reset()

        # 初始化chasersat位置
        self._chasersat_position = [0, 0, 0]
        physics.named.data.qpos['chasersat:joint'][0] = self._chasersat_position[0]
        physics.named.data.qpos['chasersat:joint'][1] = self._chasersat_position[1]
        physics.named.data.qpos['chasersat:joint'][2] = self._chasersat_position[2]

        # 初始化targetsat位置
        x_pos = self.random.uniform(3.0, 6.0, size=1)
        y_pos = self.random.uniform(-2.0, 2.0, size=1)
        z_pos = self.random.uniform(-2.0, 2.0, size=1)
        self._target_position = [x_pos, y_pos, z_pos]
        # self._target_position = self.random.uniform(1.5, 6.0, size=3)
        # self._target_position = [6, 0, 0]
        physics.named.data.qpos['targetsat:joint'][0] = self._target_position[0].item() if hasattr(self._target_position[0], 'item') else self._target_position[0]
        physics.named.data.qpos['targetsat:joint'][1] = self._target_position[1].item() if hasattr(self._target_position[1], 'item') else self._target_position[1]
        physics.named.data.qpos['targetsat:joint'][2] = self._target_position[2].item() if hasattr(self._target_position[2], 'item') else self._target_position[2]

        # 重置机械臂关节位置到中间值
        arm_joints = [
            'arm:shoulder_pan_joint',
            'arm:shoulder_lift_joint',
            'arm:elbow_joint',
            'arm:wrist_1_joint',
            'arm:wrist_2_joint',
            'arm:wrist_3_joint'
        ]

        for joint in arm_joints:
            physics.named.data.qpos[joint][:] = self.random.uniform(-0.1, 0.1)
            # physics.named.data.qpos[joint][:] = self.random.uniform(-0.5, 0.5)

    def get_observation(self, physics):
        """构建观测空间"""
        # obs = OrderedDict()
        obs = {}

        # chasersat状态
        obs['chasersat_pos'] = physics.data.body(self._chasersat_body_id).xpos.copy()
        obs['chasersat_xmat'] = physics.data.body(self._chasersat_body_id).xmat.copy()
        obs['chasersat_lin_vel'] = physics.data.body(self._chasersat_body_id).cvel[3:].copy()
        obs['chasersat_ang_vel'] = physics.data.body(self._chasersat_body_id).cvel[:3].copy()

        # targetsat状态
        obs['targetsat_pos'] = physics.data.body(self._targetsat_body_id).xpos.copy()
        obs['targetsat_xmat'] = physics.data.body(self._targetsat_body_id).xmat.copy()
        
        # 相对状态
        obs['target_rel_pos'] = obs['targetsat_pos'] - obs['chasersat_pos']

        # 机械臂状态
        arm_joints = [
            'arm:shoulder_pan_joint',
            'arm:shoulder_lift_joint',
            'arm:elbow_joint',
            'arm:wrist_1_joint',
            'arm:wrist_2_joint',
            'arm:wrist_3_joint'
        ]
        arm_joint_pos = []
        arm_joint_vel = []
        for joint in arm_joints:
            joint_id = physics.model.joint(joint).id
            qpos_addr = physics.model.jnt_qposadr[joint_id]
            qvel_addr = physics.model.jnt_dofadr[joint_id]
            arm_joint_pos.append(physics.data.qpos[qpos_addr])
            arm_joint_vel.append(physics.data.qvel[qvel_addr])
        obs['arm_joint_pos'] = np.array(arm_joint_pos)
        obs['arm_joint_vel'] = np.array(arm_joint_vel)

        # chasersat gripper position
        obs['gripper_pos'] = physics.data.body(self._gripper_body_id).xpos.copy()

        # targetsat handle position
        obs['handle_pos'] = physics.data.body(self._handle_body_id).xpos.copy()

        # 获取相机图像
        width = 84
        height = 84
        rgb_image = physics.render(width, height, camera_id=self._camera_id)
        obs['image_obs'] = rgb_image

        # 获取相机姿态
        obs['camera_xmat'] = physics.data.camera(self._camera_id).xmat.copy()

        # test image output
        # plt.imshow(rgb_image)
        # plt.title("Camera View")
        # plt.axis("off")
        # plt.show()

        # get necessary observation back
        keys_to_vector = ['camera_xmat', 'chasersat_xmat', 'arm_joint_pos', 'gripper_pos', 'handle_pos']
        vector_obs_list = [np.array(obs[key]).flatten() for key in keys_to_vector]
        vector_obs = np.concatenate(vector_obs_list)
        filtered_obs = {
            'image_obs': obs['image_obs'],
            'vector_obs': vector_obs
        }

        return filtered_obs

    def get_reward(self, physics):
        # 计算奖励函数
        # 位置误差奖励
        gripper_pos = physics.data.body(self._gripper_body_id).xpos.copy()
        handle_pos = physics.data.body(self._handle_body_id).xpos.copy()
        pos_error = np.linalg.norm(gripper_pos - handle_pos) / self.init_dist

        w1, w2 = 5.0, 1.0
        pos_reward = w1 * np.exp(-w1 * pos_error**2) - w2

        # velocity penalty
        w3 = 0.001
        vel_reward = -w3 * np.sum(np.abs(physics.data.qvel.copy()))

        # 控制惩罚奖励
        w4 = 0.02
        ctrl_reward = -w4 * np.log(np.sum(np.square(physics.data.ctrl.copy())))

        # print("pos_reward: ", pos_reward)
        # print("vel_reward: ", vel_reward)
        # print("ctrl_reward: ", ctrl_reward)

        # 计算总奖励
        reward = pos_reward + vel_reward + ctrl_reward

        # reward shaping
        k = 5.0
        potential = - (1 - np.exp(-k * pos_error**2))

        if not hasattr(self, 'potential_previous'):
            self.potential_previous = potential
        
        reward = reward + 0.99 * potential - self.potential_previous
        self.potential_previous = potential
        # print("potential_previous: ", self.potential_previous)
        # print("potential: ", self.potential)

        # 抓捕成功获得额外奖励
        if self._is_grasp_successful(physics):
            reward += 10.0

        if self._check_collision(physics):
            reward -= 2.0

        return reward

    def before_step(self, action, physics):
        action = action
        action = np.clip(action, -1, 1)
        forces = action[:3] * 25 # 线性力
        torques = action[3:6] * 10 # 角速度力矩

        # cw方程实现
        x, y, z = physics.data.body(self._targetsat_body_id).xpos.copy() - physics.data.body(self._chasersat_body_id).xpos.copy()
        x_dot, y_dot, z_dot = physics.data.body(self._targetsat_body_id).cvel[3:].copy() - physics.data.body(self._chasersat_body_id).cvel[3:].copy()
        # print("x, y, z, x_dot, y_dot, z_dot: ", x, y, z, x_dot, y_dot, z_dot)

        x_ddot = 3 * self.n**2 * x + 2 * self.n * y_dot
        y_ddot = -2 * self.n * x_dot
        z_ddot = -self.n**2 * z

        force_x = self.chasersat_mass * x_ddot
        force_y = self.chasersat_mass * y_ddot
        force_z = self.chasersat_mass * z_ddot

        # 设置控制输入
        physics.named.data.ctrl['x_force'] = forces[0] + force_x
        physics.named.data.ctrl["y_force"] = forces[1] + force_y
        physics.named.data.ctrl["z_force"] = forces[2] + force_z
        physics.named.data.ctrl["x_torque"] = torques[0]
        physics.named.data.ctrl["y_torque"] = torques[1]
        physics.named.data.ctrl["z_torque"] = torques[2]
        
        arm_joints = [
            "arm:shoulder_pan_T",
            "arm:shoulder_lift_T",
            "arm:elbow_T",
            "arm:wrist_1_T",
            "arm:wrist_2_T",
            "arm:wrist_3_T"
        ]
        # 将[-1,1]的动作范围映射到关节控制范围
        joint_positions = []
        joint_ranges = [2.0942, 2.0942, 2.0942, 3.14, 3.14, 3.14]
        
        for i, joint in enumerate(arm_joints):
            position = action[i + 6] * joint_ranges[i]
            joint_positions.append(position)
            physics.named.data.ctrl[joint] = position

    def after_step(self, physics):
        gripper_pos = physics.data.body(self._gripper_body_id).xpos.copy()
        handle_pos = physics.data.body(self._handle_body_id).xpos.copy()
        pos_error = np.linalg.norm(gripper_pos - handle_pos)
        self.potential = -np.log(pos_error / self.init_dist)

    def get_termination(self, physics):
        if self._is_grasp_successful(physics) or self._check_collision(physics):
            return 1.0

        return None

    def _check_collision(self, physics):
        for i in range(physics.data.ncon):
            contact = physics.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            geom1_body = physics.model.geom_bodyid[geom1_id]
            geom2_body = physics.model.geom_bodyid[geom2_id]

            if (geom1_body == self._targetsat_body_id and geom2_body == self._chasersat_body_id) or \
                    (geom2_body == self._targetsat_body_id and geom1_body == self._chasersat_body_id):
                return True

        return False

    def _is_grasp_successful(self, physics):
        gripper_pos = physics.data.body(self._gripper_body_id).xpos.copy()
        handle_pos = physics.data.body(self._handle_body_id).xpos.copy()
        distance = np.linalg.norm(gripper_pos - handle_pos)

        grasp_threshold = 0.5

        if distance < grasp_threshold:
            print("*****************************************************")
            print("********** Successful grasp the targetsat! **********")
            print("*****************************************************")
            return True

        return False
    

    # def _create_observation_space(self):
    #     observation_spec = self.env.observation_spec()

    #     image_obs_spec = observation_spec['image_obs']
    #     vector_obs_spec = observation_spec['vector_obs']

    #     low = np.zeros(image_obs_spec.shape, dtype=np.uint8)
    #     high = np.full(image_obs_spec.shape, 255, dtype=np.uint8)
    #     vec_low = np.full(vector_obs_spec.shape, -np.inf, dtype=np.float32)
    #     vec_high = np.full(vector_obs_spec.shape, np.inf, dtype=np.float32)

    #     observation_space = spaces.Dict({
    #         'image_obs': spaces.Box(low=low, high=high, dtype=np.uint8),
    #         'vector_obs': spaces.Box(low=vec_low, high=vec_high, dtype=np.float32),
    #     })

    #     return observation_space
    
    # def _create_action_space(self):
    #     action_spec = self.env.action_spec()
    #     low = np.full(action_spec.shape, -1)
    #     high = np.full(action_spec.shape, 1)
    #     action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    #     return action_space


class SpaceRobotEnv(control.Environment):
    def __init__(self, xml_path=_XML_PATH, time_limit=_DEFAULT_TIME_LIMIT,
                 control_timestep=_CONTROL_TIMESTEP,
                 n_sub_steps=None,
                 flat_observation=False,
                 legacy_step=False):

        """
        初始化 SpaceRobotEnv 环境。

        Args:
            xml_path (str): MuJoCo 模型 XML 文件的路径。
            time_limit (float): 每个 episode 的最大时长（秒）。
            control_timestep (float): 控制步长（秒）。
            n_sub_steps (int): 每个控制步执行的仿真子步数。
            flat_observation (bool): 是否将观测展平成一维数组（True/False）。
            legacy_step (bool): 是否采用 legacy_step 语义（通常设置为 False）。
        """

        physics = mujoco.Physics.from_xml_path(xml_path)
        task = SpaceRobot(physics)

        super(SpaceRobotEnv, self).__init__(physics=physics, task=task, time_limit=time_limit,
                         control_timestep=control_timestep, n_sub_steps=n_sub_steps,
                         flat_observation=flat_observation, legacy_step=legacy_step)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='image_obs'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)
    

class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    
def make(name, frame_stack, action_repeat, seed):
    env = SpaceRobotEnv(xml_path=_XML_PATH, time_limit=_DEFAULT_TIME_LIMIT,
                         control_timestep=_CONTROL_TIMESTEP,
                         n_sub_steps=None,
                         flat_observation=False)
    pixels_key = 'image_obs'
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    return env


def random_policy(timestep):
    del timestep
    return np.random.uniform(low=-1, high=1, size=env.action_spec().shape)


if __name__ == "__main__":
    # 创建环境
    env = make("SpaceRobot", frame_stack=4, action_repeat=1, seed=42)
    timestep = env.reset()
    # viewer.launch(env, policy=random_policy)

    print("action_spec: ", env.action_spec())
    print("observation_spec: ", env.observation_spec())

    while not timestep.last():
        action = random_policy(timestep)
        timestep = env.step(action)
        print("Action: ", action)
        # print("Observation: ", timestep.observation)
        print("Reward: ", timestep.reward)
        print(timestep.last())
    env.close()
    
    