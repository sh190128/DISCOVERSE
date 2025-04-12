import numpy as np
import gymnasium
import mujoco
from gymnasium import spaces
from discoverse.examples.tasks_mmk2.pick_kiwi import SimNode, cfg
from discoverse.task_base import MMK2TaskBase
from discoverse.utils import get_body_tmat
from skimage.transform import resize
import os
import cv2


class Env(gymnasium.Env):
    def __init__(self, task_base=None, render=False, use_gaussian=False):
        super(Env, self).__init__()

        # 环境配置
        cfg.use_gaussian_renderer = use_gaussian  # 关闭高斯渲染器
        cfg.init_key = "pick"  # 初始化模式为"抓取"
        cfg.gs_model_dict["plate_white"] = "object/plate_white.ply"  # 定义白色盘子的模型路径
        cfg.gs_model_dict["kiwi"] = "object/kiwi.ply"  # 定义奇异果的模型路径
        cfg.gs_model_dict["background"] = "scene/tsimf_library_1/point_cloud.ply"  # 定义背景的模型路径
        cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_kiwi.xml"  # MuJoCo 环境文件路径
        cfg.obj_list = ["plate_white", "kiwi"]  # 环境中包含的对象列表
        cfg.sync = True  # 是否同步更新
        cfg.headless = not render  # 根据render参数决定是否显示渲染画面
        cfg.render_set  = {
                                    "fps"    : 5,
                                    "width"  : 640,
                                    "height" : 480
                                }
        # 创建基础任务环境
        if task_base is None:
            self.task_base = MMK2TaskBase(cfg)  # 使用给定配置初始化基础任务环境
        else:
            self.task_base = task_base
        self.mj_model = self.task_base.mj_model  # 获取MuJoCo模型
        self.mj_data = self.task_base.mj_data  # 获取MuJoCo数据

        # 动作空间：机械臂关节角度控制
        # 使用actuator_ctrlrange来确定动作空间范围
        ctrl_range = self.mj_model.actuator_ctrlrange  # 获取控制器范围
        
        # 定义一个不受限制的动作空间，让算法可以探索全部自由度
        # 使用一个非常大的范围来模拟无限制
        unlimited_low = np.ones_like(ctrl_range[:, 0]) * -10.0  # 设置较大的负值
        unlimited_high = np.ones_like(ctrl_range[:, 1]) * 10.0   # 设置较大的正值
        
        self.action_space = spaces.Box(  # 定义动作空间
            low=unlimited_low,  # 使用无限制的下限
            high=unlimited_high,  # 使用无限制的上限
            dtype=np.float32
        )
        
        # 保存原始控制范围，用于在step函数中对动作进行裁剪，防止损坏机器人
        self.original_ctrl_range = ctrl_range.copy()

        # 观测空间：基于视觉的观察空间
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(3, 84, 84),  # RGB图像，调整为84x84大小
            dtype=np.float32
        )

        self.max_steps = 1000  # 最大时间步数
        self.current_step = 0  # 当前时间步数

        # 初始化奖励信息字典
        self.reward_info = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0  # 重置当前时间步数

        try:
            # 重置环境
            self.task_base.reset()  # 重置任务环境
            # self.task_base.domain_randomization()  # 域随机化
            img = self.task_base.getObservation()["img"][0]
            # 保存原始图像
            cv2.imwrite('original.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            observation = self._get_obs()  # 获取初始观测
            info = {}
            return observation, info  # 返回观察值和信息
        except Exception as e:
            print(f"重置环境失败: {str(e)}")
            raise e

    def step(self, action):
        try:
            self.current_step += 1  # 更新当前时间步数

            # 执行动作
            # 确保动作的形状正确
            action_array = np.array(action, dtype=np.float32)
            if action_array.shape != self.action_space.shape:
                # 修改为不打印警告，而是自动修复
                if len(action_array.shape) > 1 and action_array.shape[0] == 1:
                    # 将 (1, 19) 形状转换为 (19,)
                    action_array = action_array.squeeze(0)
                else:
                    # 如果仍然不匹配，则记录一次
                    if not hasattr(self, '_action_shape_reported'):
                        print(f"动作形状不匹配: 期望 {self.action_space.shape}, 实际 {action_array.shape}，已自动调整")
                        self._action_shape_reported = True

            # 将动作限制在合法范围内
            clipped_action = np.clip(
                action_array,
                self.action_space.low,
                self.action_space.high
            )

            # 为了确保物理模拟的稳定性，在传递给物理引擎前再次裁剪到原始控制范围
            # 这样可以让RL算法探索更大的动作空间，但不会导致模拟器崩溃
            engine_safe_action = np.clip(
                clipped_action,
                self.original_ctrl_range[:, 0],
                self.original_ctrl_range[:, 1]
            )

            # # 直接更新控制信号，不通过task_base
            # self.mj_data.ctrl[:] = engine_safe_action  # 更新控制器信号
            # mujoco.mj_step(self.mj_model, self.mj_data)  # 模拟物理引擎一步

            self.task_base.step(engine_safe_action)

            # 获取新的状态
            observation = self._get_obs()  # 获取新的观察值
            reward = self._compute_reward()  # 计算奖励
            terminated = self._check_termination()  # 检查是否终止
            truncated = self.current_step >= self.max_steps  # 检查是否超出最大步数
            info = {}  # 信息字典

            # 将奖励信息添加到info中
            info.update(self.reward_info)
            return observation, reward, terminated, truncated, info
        except Exception as e:
            # 仅在第一次出现时打印详细错误
            if not hasattr(self, '_step_error_reported'):
                print(f"执行动作失败: {str(e)}")
                self._step_error_reported = True
            # 发生错误时，返回一个默认观察值和适当的终止信号
            observation = np.zeros((3, 84, 84), dtype=np.float32)
            return observation, 0.0, False, True, {}  # 使用truncated=True表示异常终止

    def _get_obs(self):
        # # 获取摄像头图像
        # action = np.zeros_like(self.mj_data.ctrl)  # 创建空动作
        # obs_dict = self.task_base.step(action)  # 获取观察字典

        # 使用getObservation()直接获取观察值，而不是通过step方法
        obs_dict = self.task_base.getObservation()  # 直接获取观察字典
        
        # 不再打印警告，直接处理不同类型的观察值
        try:
            # 处理元组类型的观察值
            if isinstance(obs_dict, tuple):
                # 如果是元组，获取第一个元素（通常包含图像数据）
                if len(obs_dict) > 0:
                    obs_element = obs_dict[0]
                    # 检查元组第一个元素的类型
                    if isinstance(obs_element, dict) and 'img' in obs_element:
                        img = obs_element['img'][0]
                    elif isinstance(obs_element, list) and len(obs_element) > 0 and 'img' in obs_element[0]:
                        img = obs_element[0]['img'][0]
                    else:
                        img = obs_element  # 尝试直接使用元素
                else:
                    # 空元组情况，返回零矩阵
                    return np.zeros((3, 84, 84), dtype=np.float32)
            # 处理字典类型的观察值
            elif isinstance(obs_dict, dict) and 'img' in obs_dict:
                img = obs_dict['img'][0]  # 直接从字典中获取
            # 处理列表类型的观察值
            elif isinstance(obs_dict, list) and len(obs_dict) > 0 and 'img' in obs_dict[0]:
                img = obs_dict[0]['img'][0]  # 从列表的第一个字典获取
            else:
                # 如果无法识别结构，静默返回零矩阵（不打印警告）
                return np.zeros((3, 84, 84), dtype=np.float32)
            
            # 图像处理逻辑保持不变
            img = img.astype(np.float32) / 255.0  # 归一化到[0,1]范围
            img = img.transpose(2, 0, 1)  # 转换为(C, H, W)格式
            img = resize(img, (3, 84, 84), anti_aliasing=True)  # 调整大小为84x84
            return img
        except Exception as e:
            # 捕获异常但不打印详细信息，避免训练过程中输出大量日志
            # 只在第一次出现时打印错误，之后静默处理
            if not hasattr(self, '_obs_error_reported'):
                print(f"处理观察值时出错: {e}")
                self._obs_error_reported = True
            # 异常情况下返回零矩阵
            return np.zeros((3, 84, 84), dtype=np.float32)

    def _compute_reward(self):
        # 获取位置信息
        tmat_kiwi = get_body_tmat(self.mj_data, "kiwi")  # 奇异果的变换矩阵
        tmat_plate = get_body_tmat(self.mj_data, "plate_white")  # 盘子的变换矩阵
        tmat_rgt_arm = get_body_tmat(self.mj_data, "rgt_arm_link6")  # 右臂末端效应器的变换矩阵

        kiwi_pos = np.array([tmat_kiwi[1, 3], tmat_kiwi[0, 3], tmat_kiwi[2, 3]])  # 奇异果的位置
        plate_pos = np.array([tmat_plate[1, 3], tmat_plate[0, 3], tmat_plate[2, 3]])  # 盘子的位置
        rgt_arm_pos = np.array([tmat_rgt_arm[1, 3], tmat_rgt_arm[0, 3], tmat_rgt_arm[2, 3]])  # 右臂末端的位置

        # 计算距离
        distance_to_kiwi = np.linalg.norm(rgt_arm_pos - kiwi_pos)  # 右臂末端到奇异果的距离
        kiwi_to_plate = np.linalg.norm(kiwi_pos - plate_pos)  # 奇异果到盘子的距离

        # 计算各种奖励
        # 接近奖励：鼓励机械臂靠近奇异果
        approach_reward = 0.0
        if distance_to_kiwi < 0.1:
            approach_reward = 10.0
        else:
            approach_reward = -distance_to_kiwi
        
        # approach_reward = 0.0
        # if distance_to_kiwi < 0.1:
        #     approach_reward = 5.0
        # else:
        #     # base_reward * exp(-distance_scale * distance)

        #     distance_scale = 5.0  # 控制衰减速度的参数
        #     approach_reward = 1.0 * np.exp(-distance_scale * distance_to_kiwi)

        # 放置奖励：鼓励机械臂将奇异果放置到盘子
        place_reward = 0.0
        if kiwi_to_plate < 0.02:  # 成功放置
            place_reward = 10.0
        elif kiwi_to_plate < 0.1:  # 比较接近
            place_reward = 2.0
        else:
            place_reward = -kiwi_to_plate

        # 步数惩罚：每一步都有一定的惩罚
        step_penalty = -0.01 * self.current_step

        # 动作幅度惩罚：惩罚较大的控制信号
        action_magnitude = np.mean(np.abs(self.mj_data.ctrl))
        action_penalty = -0.1 * action_magnitude
        
        # 添加探索奖励：鼓励尝试不同的动作
        # 如果当前动作与安全范围边界接近，给予额外奖励
        action_diff_to_boundary = np.minimum(
            np.abs(self.mj_data.ctrl - self.original_ctrl_range[:, 0]),
            np.abs(self.mj_data.ctrl - self.original_ctrl_range[:, 1])
        )
        # 当动作接近边界时，给予探索奖励
        exploration_reward = 0.1 * np.sum(np.exp(-10.0 * action_diff_to_boundary))

        # 总奖励
        total_reward = (
                approach_reward +
                place_reward +
                step_penalty +
                action_penalty +
                exploration_reward
        )

        # 记录详细的奖励信息供日志使用
        self.reward_info = {
            "rewards/total": total_reward,
            "rewards/approach": approach_reward,
            "rewards/place": place_reward,
            "rewards/step_penalty": step_penalty,
            "rewards/action_penalty": action_penalty,
            "rewards/exploration": exploration_reward,
            "info/distance_to_kiwi": distance_to_kiwi,
            "info/kiwi_to_plate": kiwi_to_plate,
            "info/action_magnitude": action_magnitude
        }

        return total_reward

    def _check_termination(self):
        # 检查是否完成任务
        tmat_kiwi = get_body_tmat(self.mj_data, "kiwi")  # 奇异果的变换矩阵
        tmat_plate = get_body_tmat(self.mj_data, "plate_white")  # 盘子的变换矩阵

        kiwi_pos = np.array([tmat_kiwi[1, 3], tmat_kiwi[0, 3], tmat_kiwi[2, 3]])  # 奇异果的位置
        plate_pos = np.array([tmat_plate[1, 3], tmat_plate[0, 3], tmat_plate[2, 3]])  # 盘子的位置

        # 如果奇异果成功放置在盘子上
        if np.linalg.norm(kiwi_pos - plate_pos) < 0.02:
            return True  # 任务完成，终止环境
        return False
