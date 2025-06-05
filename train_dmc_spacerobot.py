import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'glfw'

from pathlib import Path

import hydra
import numpy as np
import utils
import torch
from dm_env import specs

import dmc_spacerobot

from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import wandb
import re
import threading
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        if self.cfg.use_wandb:
            exp_name = '_'.join([cfg.task_name, str(cfg.agent._target_.split('.')[-1])])
            group_name = re.search(r'\.(.+)\.', cfg.agent._target_).group(1)
            
            # 创建一个可序列化的配置字典
            config_dict = {}
            for k, v in vars(cfg).items():
                if not k.startswith('_') and not callable(v):
                    try:
                        # 尝试简单转换为字符串检查可序列化性
                        str(v)
                        config_dict[k] = v
                    except:
                        config_dict[k] = str(v)

            wandb.init(project="DrM",
                    group=group_name,
                    name=exp_name,
                    config=config_dict)
            
            # wandb.init(project="DrM",
            #            group=group_name,
            #            name=exp_name,
            #            config=cfg)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self._discount = cfg.discount
        self._nstep = cfg.nstep

        # 共享计数器和锁
        self._total_env_steps = 0 # 记录总的环境交互步数
        self._total_env_steps_lock = threading.Lock() # 用于保护 _total_env_steps 的锁
        self._global_episode = 0 # 记录全局回合数
        self._global_episode_lock = threading.Lock() # 用于保护 _global_episode 的锁

        # 从配置中获取收集器线程数，默认为1
        self.num_collector_threads = getattr(self.cfg, 'num_collector_threads', 1)
        self.collector_threads = [] # 存储收集器线程的列表
        self.stop_event = threading.Event() # 用于通知所有线程停止的事件
        # self.log_queue = queue.Queue() if self.num_collector_threads > 1 else None # 可选：用于线程间日志记录

        # 在创建智能体之前需要设置环境以获取观测和动作空间规范
        self.setup_env_and_replay() # 重构了部分 setup 逻辑

        # 创建智能体，使用 spec_holder 环境获取规范
        self.agent = make_agent(self.train_env_spec_holder.observation_spec(),
                                self.train_env_spec_holder.action_spec(), self.cfg.agent)
        
        self.timer = utils.Timer()
        self._agent_update_steps = 0 # 记录智能体更新的次数

    def setup_env_and_replay(self):
        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=self.cfg.use_tb,
                             use_wandb=self.cfg.use_wandb)
        # create envs
        self.train_env_spec_holder = dmc_spacerobot.make(self.cfg.frame_stack, self.cfg.action_repeat)
        # self.eval_env = dmc_spacerobot.make(self.cfg.frame_stack, self.cfg.action_repeat)

        # create replay buffer
        data_specs = (self.train_env_spec_holder.observation_spec(),
                      self.train_env_spec_holder.action_spec(),
                      specs.Array((1, ), np.float32, 'reward'),
                      specs.Array((1, ), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')
        # 创建经验回放加载器，用于从存储中批量加载数据
        self.replay_loader, self.buffer = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers, self.cfg.save_buffer,
            self._nstep,
            self._discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        # 训练视频记录器：即使在多线程模式下，也创建它，后续会传递给指定线程
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None
        )
        
        # 如果主要使用收集器线程，则关闭用于获取规范的环境实例
        if self.num_collector_threads > 1:
            self.train_env_spec_holder.close()
            self.train_env_spec_holder = None # 明确释放
        else: # 在单线程模式下，这个实例是主要的训练环境
            self.train_env = self.train_env_spec_holder

    # 线程安全地获取和增加共享计数器的方法
    def get_total_env_steps(self):
        with self._total_env_steps_lock:
            return self._total_env_steps

    def increment_total_env_steps(self):
        with self._total_env_steps_lock:
            self._total_env_steps += 1
            return self._total_env_steps

    def get_global_episode(self):
        with self._global_episode_lock:
            return self._global_episode
            
    def increment_global_episode(self):
        with self._global_episode_lock:
            self._global_episode += 1
            return self._global_episode            
    @property
    def global_step(self):
        return self.get_total_env_steps()

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
         
        # 在评估开始时创建评估环境
        eval_env = dmc_spacerobot.make(self.cfg.frame_stack, self.cfg.action_repeat)
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.get_global_episode())
            log('step', self.global_step)

    def train(self):
        # predicates 现在使用 self.global_step (即 self.get_total_env_steps())
        # 注意：这些谓词的计数是总的环境交互步数，而不是智能体更新的步数。
        train_until_step = utils.Until(self.cfg.num_train_frames // self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames // self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames // self.cfg.action_repeat)
        time_step = None
        episode_step, episode_reward = 0, 0
        
        metrics = None # 在循环外初始化 metrics

         # 启动收集器线程
        if self.num_collector_threads > 1:
            print(f"正在启动 {self.num_collector_threads} 个数据收集器线程...")
            for i in range(self.num_collector_threads):
                video_recorder_for_thread = None
            # 如果是线程0，并且配置了保存训练视频，并且记录器已创建，则传递记录器
                if i == 0 and self.cfg.save_train_video and self.train_video_recorder is not None:
                    video_recorder_for_thread = self.train_video_recorder

                # 创建并启动 DataCollector 线程
                collector = DataCollector(i, self.cfg, self.agent, self.replay_storage, self, 
                                          self.stop_event, self.cfg.num_seed_frames // self.cfg.action_repeat,
                                          video_recorder=video_recorder_for_thread) # <--- 确保传递 video_recorder
                collector.start()
                self.collector_threads.append(collector)

        else: # 单线程数据收集 (原始行为，但针对新的步数计数进行了修改)
            print("正在以单线程数据收集模式启动...")

            if not hasattr(self, 'train_env'):
                 # 这是一个防御性检查，理论上 setup_env_and_replay 应该已经处理了
                 raise RuntimeError("错误：在单线程模式下 train_env 未被初始化！")
            time_step = self.train_env.reset() # 现在可以安全调用
            self.replay_storage.add(time_step)
            if self.cfg.save_train_video and self.train_video_recorder:
                self.train_video_recorder.init(time_step.observation)      
        

         
        while train_until_step(self.global_step):
            # 单线程数据收集逻辑
            '''
            if self.num_collector_threads <= 1:
                if time_step.last():
                    self._global_episode += 1
                    self.train_video_recorder.save(f'{self.global_frame}.mp4')
                    # wait until all the metrics schema is populated
                    if metrics is not None:
                    # log stats
                        elapsed_time, total_time = self.timer.reset()
                        episode_frame = episode_step * self.cfg.action_repeat
                        with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                            log('fps', episode_frame / elapsed_time)
                            log('total_time', total_time)
                            log('episode_reward', episode_reward)
                            log('episode_length', episode_frame)
                            log('episode', self.global_episode)
                            log('buffer_size', len(self.replay_storage))
                            log('step', self.global_step)

                    # reset env
                    time_step = self.train_env.reset()
                    self.replay_storage.add(time_step)
                    self.train_video_recorder.init(time_step.observation)
                    # try to save snapshot
                    if self.cfg.save_snapshot:
                        self.save_snapshot()
                    episode_step = 0
                    episode_reward = 0 ##

                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=False)
                next_time_step = self.train_env.step(action)
                self.increment_total_env_steps() # 单线程模式下手动增加总步数
                
                episode_reward += next_time_step.reward
                self.replay_storage.add(next_time_step)
                # self.train_video_recorder.record(next_time_step.observation)
                episode_step += 1
                time_step = next_time_step
            '''
            if self.num_collector_threads <= 1: # 单线程数据收集和视频录制
                if time_step is None: 
                    # 如果是单线程模式但 time_step 是 None，说明初始化逻辑有问题
                    # 或者代码逻辑在不期望的情况下进入了这个分支
                    raise RuntimeError("错误：在单线程训练循环中 time_step 未被正确初始化。")
                
                if time_step.last(): # type: ignore
                    self.increment_global_episode() # 使用线程安全的方法
                    if self.cfg.save_train_video and self.train_video_recorder:
                        self.train_video_recorder.save(f'{self.global_frame}_train_main.mp4')
                    
                    if metrics is not None:
                        elapsed_time, total_time = self.timer.reset()
                        episode_frame = episode_step * self.cfg.action_repeat # type: ignore
                        with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                            log('fps', episode_frame / elapsed_time)
                            log('total_time', total_time)
                            log('episode_reward', episode_reward) # type: ignore
                            log('episode_length', episode_frame)
                            log('episode', self.get_global_episode()) # 使用线程安全的方法
                            log('buffer_size', len(self.replay_storage))
                            log('step', self.global_step)

                    time_step = self.train_env.reset()
                    self.replay_storage.add(time_step)
                    if self.cfg.save_train_video and self.train_video_recorder:
                        self.train_video_recorder.init(time_step.observation)
                    if self.cfg.save_snapshot:
                        self.save_snapshot()
                    episode_step, episode_reward = 0, 0

                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation, # type: ignore
                                            self.global_step,
                                            eval_mode=False)
                next_time_step = self.train_env.step(action)
                self.increment_total_env_steps()
                
                episode_reward += next_time_step.reward # type: ignore
                self.replay_storage.add(next_time_step)
                if self.cfg.save_train_video and self.train_video_recorder:
                    self.train_video_recorder.record(next_time_step.observation)
                episode_step += 1 # type: ignore
                time_step = next_time_step

            # --- 评估和智能体更新逻辑 ---    
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            if not seed_until_step(self.global_step) and \
               len(self.replay_storage) >= self.cfg.batch_size :
                if self._agent_update_steps % self.cfg.update_every_steps == 0:
                    metrics = self.agent.update(self.replay_iter, self.global_step)
                else:
                    metrics = dict()
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
                self._agent_update_steps += 1
            
            # 定期保存快照的逻辑可以保留或调整
            if self.cfg.save_snapshot and self._agent_update_steps > 0 and \
               (self._agent_update_steps % getattr(self.cfg, 'save_snapshot_every_updates', 1000) == 0): # 例如每N次更新保存一次
                self.save_snapshot()

            # 如果经验池数据不足，短暂休眠，避免CPU空转
            if len(self.replay_storage) < self.cfg.batch_size and not seed_until_step(self.global_step):
                import time # 确保导入 time
                time.sleep(0.01)
           
            # 可选：处理来自工作线程的日志 (如果使用了 self.log_queue)
            # while self.log_queue and not self.log_queue.empty():
            #    log_item = self.log_queue.get_nowait()
            #    # ... 处理 log_item, 更新主日志记录器 ...


            # sample action
#            with torch.no_grad(), utils.eval_mode(self.agent):
#                action = self.agent.act(time_step.observation,
#                                        self.global_step,
#                                        eval_mode=False)

            # try to update the agent
            # 更新智能体
            # 确保经验回放池中有足够的样本并且已过种子阶段 (seed phase)
            if not seed_until_step(self.global_step) and \
               len(self.replay_storage) >= self.cfg.batch_size :# 检查缓冲区是否有足够的样本
                # 根据原始逻辑，每隔 update_every_steps 更新一次智能体
                if self._agent_update_steps % self.cfg.update_every_steps == 0:
                    metrics = self.agent.update(self.replay_iter, self.global_step) # 传递当前总环境步数
                else:
                    metrics = dict() # 如果不到更新的时候，metrics 为空字典
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
                self._agent_update_steps += 1 # 增加智能体更新次数的计数
            # 如果使用多线程且缓冲区数据不够快速填充以供更新，则短暂休眠以避免忙等待
            if self.num_collector_threads > 1 and len(self.replay_storage) < self.cfg.batch_size :
                import time # 确保导入 time 模块
                time.sleep(0.01)

#               metrics = self.agent.update(
#                    self.replay_iter, self.global_step
#                ) if self.global_step % self.cfg.update_every_steps == 0 else dict(
#                )
#                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
#           time_step = self.train_env.step(action)
#           episode_reward += time_step.reward
#           self.replay_storage.add(time_step)
#           self.train_video_recorder.record(time_step.observation)
#           episode_step += 1
#            self._global_step += 1

        # 清理：通知并等待所有收集器线程结束
        print("正在停止收集器线程...")
        self.stop_event.set() # 设置停止事件，通知所有线程退出循环
        for collector in self.collector_threads:
            collector.join() # 等待线程执行完毕
            print(f"收集器线程 {collector.thread_id} 已结束。")

        # 关闭在单线程模式下使用的训练环境
        ''' 
        if self.num_collector_threads <= 1 and hasattr(self, 'train_env') and self.train_env : # 检查 train_env 是否存在
            self.train_env.close()
        elif self.num_collector_threads > 1 and self.train_env_spec_holder : # 如果是多线程，且spec_holder存在
             self.train_env_spec_holder.close()
        '''
        if hasattr(self, 'train_env') and self.train_env and self.num_collector_threads <= 1 :
            self.train_env.close()
        if self.eval_env: # 关闭评估环境
            self.eval_env.close()
        print("训练结束。")

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_agent_update_steps', '_total_env_steps','_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save if k in self.__dict__}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

class DataCollector(threading.Thread):
    def __init__(self, thread_id, cfg, agent, replay_storage, workspace_ref, stop_event, num_expl_steps, video_recorder=None):
        super().__init__()
        self.thread_id = thread_id
        self.cfg = cfg
        self.agent = agent  # 共享的智能体实例
        self.replay_storage = replay_storage # 共享的经验回放池
        self.workspace_ref = workspace_ref # 对主 Workspace 实例的引用
        self.stop_event = stop_event # 用于通知线程停止的事件
        self.num_expl_steps = num_expl_steps # 来自 cfg.num_seed_frames，用于探索
        self.video_recorder = video_recorder

        # 每个收集器拥有自己的环境实例
        self.train_env = dmc_spacerobot.make(self.cfg.frame_stack, self.cfg.action_repeat)
        # self.log_queue = self.workspace_ref.log_queue # 可选：用于从工作线程记录日志

    def run(self):
        episode_step = 0
        episode_reward = 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step) # add 方法已修改为线程安全

        # 如果此线程被指定为录制线程并且 video_recorder 存在
        if self.video_recorder:
            self.video_recorder.init(time_step.observation) # 初始化录制器

        while not self.stop_event.is_set():
            # 获取当前的全局环境交互总步数，用于智能体的探索策略
            current_total_env_steps = self.workspace_ref.get_total_env_steps()

            with torch.no_grad(), utils.eval_mode(self.agent):
                # agent.act 方法使用 'step' 参数进行探索调度
                action = self.agent.act(time_step.observation,
                                        current_total_env_steps,
                                        eval_mode=False) # eval_mode=False 表示允许探索

            next_time_step = self.train_env.step(action)
            self.workspace_ref.increment_total_env_steps() # 更新总步数

            # 如果此线程负责视频录制，则记录当前帧
            if self.video_recorder:
                self.video_recorder.record(next_time_step.observation)

            episode_reward += next_time_step.reward
            self.replay_storage.add(next_time_step) # add 方法已修改为线程安全
            episode_step += 1

            if next_time_step.last(): # 如果一个回合结束
                self.workspace_ref.increment_global_episode() # 线程安全地增加全局回合数
                # 可选：通过 workspace 或共享队列记录回合统计信息
                # if self.log_queue:
                #     self.log_queue.put({'type': 'episode_end', 'thread_id': self.thread_id,
                #                           'reward': episode_reward, 'length': episode_step * self.cfg.action_repeat})
                
                # 如果此线程负责视频录制，则保存视频
                if self.video_recorder:
                    # 使用 workspace_ref.global_frame 确保文件名与评估视频一致
                    video_filename = f'{self.workspace_ref.global_frame}_train_thread{self.thread_id}.mp4' # 可以加入线程ID以区分
                    self.video_recorder.save(video_filename)

                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)

                episode_step = 0
                episode_reward = 0
            else:
                time_step = next_time_step
        
        self.train_env.close() # 关闭环境

@hydra.main(config_path='cfgs', config_name='config_spacerobot')
def main(cfgs):
    from train_dmc_spacerobot import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfgs)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()