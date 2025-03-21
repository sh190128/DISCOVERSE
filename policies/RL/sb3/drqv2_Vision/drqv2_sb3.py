import warnings
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import get_schedule_fn, update_learning_rate
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

import utils

SelfDrQV2 = TypeVar("SelfDrQV2", bound="DrQV2")


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Policy(BasePolicy):
    """
    DrQV2策略类，整合编码器、Actor和Critic
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        learning_rate: float,  # 使用固定学习率
        feature_dim: int = 50,
        hidden_dim: int = 1024,
        stddev_schedule: str = "linear(1.0, 0.1, 100000)",
        stddev_clip: float = 0.3,
        *args,
        **kwargs,
    ):
        # 移除不必要的参数
        kwargs.pop('use_sde', None)  # 移除use_sde参数

        super().__init__(
            observation_space,
            action_space,
            *args,
            **kwargs,
        )
        
        # 获取观察空间形状
        obs_shape = observation_space.shape
        action_shape = action_space.shape
        
        # 初始化网络组件
        self.encoder = Encoder(obs_shape)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim)
        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 数据增强
        self.aug = RandomShiftsAug(pad=4)
        
        # 动作探索参数
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        
        # 优化器
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=learning_rate)  # 使用固定学习率
        
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播，获取动作、状态值和日志概率
        """
        obs = self.encoder(obs)
        stddev = utils.schedule(self.stddev_schedule, 0)  # 在推理时使用基础stddev
        dist = self.actor(obs, stddev)
        
        if deterministic:
            actions = dist.mean
        else:
            actions = dist.sample(clip=self.stddev_clip)
            
        log_prob = dist.log_prob(actions).sum(dim=-1)
        q1, q2 = self.critic(obs, actions)
        values = torch.min(q1, q2)
        
        return actions, values, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定观察和动作的值函数、日志概率和熵
        """
        obs = self.encoder(obs)
        stddev = utils.schedule(self.stddev_schedule, 0)
        dist = self.actor(obs, stddev)
        
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        q1, q2 = self.critic(obs, actions)
        values = torch.min(q1, q2)
        
        return values, log_prob, entropy
    
    def get_distribution(self, obs: torch.Tensor):
        """
        获取动作分布
        """
        obs = self.encoder(obs)
        stddev = utils.schedule(self.stddev_schedule, 0)
        dist = self.actor(obs, stddev)
        return dist
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        实现抽象方法 _predict，用于预测动作
        """
        try:
            # 确保观察值形状正确
            if observation is None:
                print("警告: 观察值为None，返回随机动作")
                return np.array([self.action_space.sample()])
            
            # 编码观察值
            obs = self.encoder(observation)
            stddev = utils.schedule(self.stddev_schedule, 0)
            dist = self.actor(obs, stddev)
            
            if deterministic:
                actions = dist.mean
            else:
                actions = dist.sample(clip=self.stddev_clip)
            
            
            # 检查动作形状
            if actions.size == 0:
                print("警告: 生成的动作是空的，返回随机动作")
                return np.array([self.action_space.sample()])
            
            return actions
        except Exception as e:
            print(f"预测动作时出错: {str(e)}，返回随机动作")
            return np.array([self.action_space.sample()])


class DrQV2Buffer(ReplayBuffer):
    """
    DrQV2的经验回放缓冲区
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)


class DrQV2(OffPolicyAlgorithm):
    """
    Data-regularized Q-learning V2 (DrQ-v2) 算法
    
    Paper: https://arxiv.org/abs/2107.09645
    
    :param policy: 使用的策略类
    :param env: 环境
    :param learning_rate: 学习率，可以是函数
    :param buffer_size: 回放缓冲区大小
    :param learning_starts: 开始学习前需要收集的时间步数
    :param batch_size: 批量大小
    :param tau: 目标网络更新的软更新参数
    :param gamma: 折扣因子
    :param train_freq: 更新模型的频率
    :param gradient_steps: 每次更新的梯度步数
    :param feature_dim: Actor和Critic中特征维度
    :param hidden_dim: Actor和Critic中隐藏层维度
    :param stddev_schedule: 探索噪声标准差调度
    :param stddev_clip: 噪声裁剪值
    :param num_expl_steps: 纯探索步数
    :param tensorboard_log: tensorboard日志位置
    :param policy_kwargs: 传递给策略的额外参数
    :param verbose: 日志级别
    :param seed: 随机种子
    :param device: 运行设备
    """
    
    def __init__(
        self,
        policy: Union[str, Type[DrQV2Policy]],
        env: Union[GymEnv, str],
        learning_rate: float = 1e-4,  # 直接使用固定学习率
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        batch_size: int = 256,
        tau: float = 0.01,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise = None,
        replay_buffer_class = None,
        replay_buffer_kwargs = None,
        optimize_memory_usage: bool = False,
        feature_dim: int = 50,
        hidden_dim: int = 1024,
        stddev_schedule: str = "linear(1.0, 0.1, 100000)",
        stddev_clip: float = 0.3,
        num_expl_steps: int = 2000,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,  # 添加这个参数
        **kwargs,  # 添加这个来捕获任何其他未知参数
    ):
        # 确保 train_freq 是 TrainFreq 类型
        if isinstance(train_freq, int):
            train_freq = TrainFreq(train_freq, TrainFrequencyUnit.STEP)
        elif isinstance(train_freq, tuple):
            train_freq = TrainFreq(*train_freq)

        super().__init__(
            policy=DrQV2Policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class or DrQV2Buffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            # 不要传递_init_setup_model，因为父类可能不接受它
        )
        
        # 确保 lr_schedule 被正确设置
        self.lr_schedule = get_schedule_fn(learning_rate)
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.num_expl_steps = num_expl_steps
        
        # 只有当_init_setup_model为True时才设置模型
        if _init_setup_model:
            self._setup_model()
        
    def _setup_model(self) -> None:
        """
        设置模型和回放缓冲区
        """
        # 创建回放缓冲区
        self.replay_buffer = DrQV2Buffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            n_envs=self.n_envs,
        )
        
        # 设置策略参数
        self.policy_kwargs = self.policy_kwargs or {}
        self.policy_kwargs.update({
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "stddev_schedule": self.stddev_schedule,
            "stddev_clip": self.stddev_clip,
        })
        
        # 创建策略实例
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            learning_rate=self.learning_rate,  # 使用固定学习率
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)
        
        # 当前步数
        self._n_calls = 0

    def learn(
        self: SelfDrQV2,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DrQV2",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDrQV2:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
    
    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        """
        训练模型
        """
        # 更新学习率
        self._update_learning_rate(self.policy.optimizer)
        
        losses = []
        actor_losses = []
        critic_losses = []
        q_values = []
        
        for _ in range(gradient_steps):
            # 从回放缓冲区采样
            replay_data = self.replay_buffer.sample(batch_size)
            
            # 获取观察，并应用数据增强
            obs = self.policy.aug(replay_data.observations.float())
            next_obs = self.policy.aug(replay_data.next_observations.float())
            
            # 编码观察
            with torch.no_grad():
                encoded_next_obs = self.policy.encoder(next_obs)
            encoded_obs = self.policy.encoder(obs)
            
            # 更新Critic（Q网络）
            with torch.no_grad():
                # 获取目标动作
                stddev = utils.schedule(self.stddev_schedule, self._n_calls)
                dist = self.policy.actor(encoded_next_obs, stddev)
                next_action = dist.sample(clip=self.stddev_clip)
                
                # 计算目标Q值
                target_q1, target_q2 = self.policy.critic_target(encoded_next_obs, next_action)
                target_v = torch.min(target_q1, target_q2)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_v
            
            # 当前Q值
            current_q1, current_q2 = self.policy.critic(encoded_obs, replay_data.actions)
            
            # 计算Critic损失
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            # 更新Actor
            stddev = utils.schedule(self.stddev_schedule, self._n_calls)
            dist = self.policy.actor(encoded_obs.detach(), stddev)
            action = dist.sample(clip=self.stddev_clip)
            
            q1, q2 = self.policy.critic(encoded_obs.detach(), action)
            q = torch.min(q1, q2)
            actor_loss = -q.mean()
            
            # 总损失
            loss = critic_loss + actor_loss
            
            # 优化
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()
            
            # 软更新目标网络
            utils.soft_update_params(
                self.policy.critic, self.policy.critic_target, self.tau
            )
            
            # 记录
            losses.append(loss.item())
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            q_values.append(q.mean().item())
        
        # 记录日志
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(losses) > 0:
            self.logger.record("train/loss", np.mean(losses))
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/critic_loss", np.mean(critic_losses))
            self.logger.record("train/q_values", np.mean(q_values))
            
        self._n_updates += gradient_steps
        
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        预测动作
        """
        self._n_calls += 1
        
        # 获取动作空间维度
        action_dim = self.action_space.shape[0]
        
        try:
            # 用于纯探索
            if self._n_calls < self.num_expl_steps and not deterministic:
                unscaled_action = np.array([
                    self.action_space.sample() for _ in range(self.n_envs)
                ])
                if len(unscaled_action.shape) == 1:
                    unscaled_action = unscaled_action.reshape(self.n_envs, -1)
                return unscaled_action, state
            
            # 使用policy预测动作
            actions, states = self.policy.predict(observation, state, episode_start, deterministic)
            
            # 确保动作不为None
            if actions is None:
                print("警告: policy.predict()返回None动作，使用随机动作代替")
                actions = np.array([
                    self.action_space.sample() for _ in range(self.n_envs)
                ])
            
            # 确保形状正确
            if len(actions.shape) == 1:
                actions = actions.reshape(self.n_envs, -1)
            
            # 验证动作维度
            if actions.shape != (self.n_envs, action_dim):
                print(f"警告: 动作形状不正确: {actions.shape}, 期望: ({self.n_envs}, {action_dim})")
                actions = np.array([
                    self.action_space.sample() for _ in range(self.n_envs)
                ]).reshape(self.n_envs, action_dim)
            
            # 最后增加对输出动作的形状处理
            # 如果是单环境使用，可以将 (1, action_dim) 转换为 (action_dim,)
            if self.n_envs == 1 and len(actions.shape) > 1 and actions.shape[0] == 1:
                actions = actions.squeeze(0)
            
            return actions, states
        except Exception as e:
            print(f"预测动作时出错: {str(e)}，使用随机动作代替")
            actions = np.array([
                self.action_space.sample() for _ in range(self.n_envs)
            ]).reshape(self.n_envs, action_dim)
            return actions, state
    
    # def _sample_action(
    #     self, 
    #     learning_starts: int,
    #     action_noise = None,
    #     n_envs: int = 1
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     采样动作
    #     """
    #     # 获取动作空间维度
    #     action_dim = self.action_space.shape[0]  # 应该是19
        
    #     try:
    #         if self.num_timesteps < learning_starts:
    #             # 确保随机动作具有正确的维度
    #             unscaled_action = np.array([
    #                 self.action_space.sample() for _ in range(n_envs)
    #             ])
    #             # 确保形状正确 (n_envs, action_dim)
    #             if len(unscaled_action.shape) == 1:
    #                 unscaled_action = unscaled_action.reshape(n_envs, -1)
    #             return unscaled_action, None
            
    #         # 使用Policy的predict方法
    #         actions, _ = self.policy.predict(
    #             self._last_obs, deterministic=False
    #         )
            
    #         # 确保动作形状正确
    #         if actions is None:
    #             print("警告: policy.predict()返回None动作，使用随机动作代替")
    #             actions = np.array([
    #                 self.action_space.sample() for _ in range(n_envs)
    #             ])
                
    #         # 确保动作是numpy数组
    #         if isinstance(actions, torch.Tensor):
    #             actions = actions.cpu().detach().numpy()
                
    #         # 确保形状正确
    #         if len(actions.shape) == 1:
    #             actions = actions.reshape(n_envs, -1)
            
    #         # 验证动作维度
    #         if actions.shape != (n_envs, action_dim):
    #             print(f"警告: 动作形状不正确: {actions.shape}, 期望: ({n_envs}, {action_dim})")
    #             actions = np.array([
    #                 self.action_space.sample() for _ in range(n_envs)
    #             ]).reshape(n_envs, action_dim)
                
    #         return actions, None
            
    #     except Exception as e:
    #         print(f"采样动作时出错: {str(e)}")
    #         actions = np.array([
    #             self.action_space.sample() for _ in range(n_envs)
    #         ]).reshape(n_envs, action_dim)
    #         return actions, None