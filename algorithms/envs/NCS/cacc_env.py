import configparser
import logging
import numpy as np
import pandas as pd
import torch
from gym.spaces import Box, Discrete
from datetime import datetime
from pathlib import Path
import os, sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))
from LLMmodules.CACC.catchup_utils import optimal_acc
from LLMmodules.model_config import apply_model_override, apply_test_mode_override

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
# sns.set_color_codes()
# 与 Agent 一致：由 run_cpp.py 的 --env 通过 CACC_ENV 控制
_cacc_env = os.environ.get('CACC_ENV', 'catchup')
_cacc_range = os.environ.get('CACC_RANGE', '').strip()
_range_suffix = f"_{_cacc_range}" if _cacc_range in ('range0', 'range3') else ''
catchup_flag = (_cacc_env == 'catchup')
RECORD_FLAG = True
print(f"cacc_env catchup_flag: {catchup_flag}, cacc_range: {_cacc_range or 'default'}")
# True for catchup, False for slowdown
if catchup_flag:
    config_path = PROJECT_ROOT / f'algorithms/envs/NCS/config/config_ma2c_nc_catchup{_range_suffix}.ini'
    TASK_FLAG = 1
else:
    config_path = PROJECT_ROOT / f'algorithms/envs/NCS/config/config_ma2c_nc_slowdown{_range_suffix}.ini'
    TASK_FLAG = 2
config = configparser.ConfigParser()
print(f"cacc_env config_path: {config_path}")
if not config.read(str(config_path)):
    raise FileNotFoundError(f"Cannot read CACC config: {config_path}")
COLLISION_WT = 5
COLLISION_HEADWAY = 10
VDIFF = 5

OPENAI_MODEL = apply_model_override(config) # "qwen3-235b-a22b", "deepseek-v3", "qwen3-32b", "gpt-4o-mini"
TEST_MODE = apply_test_mode_override(config) # DEBUG or EXP
if RECORD_FLAG == True:
    timestamp = datetime.now().strftime("%m%d")
    count = 0
    folder_path = PROJECT_ROOT / f"experiments/RewardTest/LLM-{OPENAI_MODEL}/{timestamp}-{TEST_MODE}"
    print(folder_path)
    # NOTE: 0312 LOW control flag
    # import pdb; pdb.set_trace()
import os

class CACCEnv:
    def __init__(self, config):
        self._load_config(config)
        self.ovm = OVMCarFollowing(self.h_s, self.h_g, self.v_max)
        self.train_mode = False
        self.cur_episode = 0
        self.is_record = False
        self.action_space = Discrete(4)
        self.observation_space=Box(-1e6, 1e6, shape=[5])
        self.reward_range=None
        self.metadata=None
        self._init_space()
        # required to achieve the same model initialization!
        np.random.seed(self.seed)

    # NOTE: acceleration
    def _constrain_speed(self, v, u):
        # apply constraints
        # 下一时刻速度 v + 加速度 * duration
        v_next = v + np.clip(u, self.u_min, self.u_max) * self.dt
        # clip速度
        v_next = np.clip(v_next, 0, self.v_max)
        # 得到真实的加速度
        u_const = (v_next - v) / self.dt
        return v_next, u_const

    # NOTE: action space
    def _get_accel(self, i, alpha, beta):
        v = self.vs_cur[i]
        h = self.hs_cur[i]
        if i:
            v_lead = self.vs_cur[i-1]
        else:
            v_lead = self.v0s[self.t]
        return self.ovm.get_accel(v, v_lead, h, alpha, beta)

    # NOTE: get rewards
    def _get_reward(self):
        '''
        Obtain reward (headway_reward + speed_reward + acc_reward + collision_reward)
        '''
        # give large penalty for collision
        if np.min(self.hs_cur) < self.h_min:
            # If collision: give the worst reward
            self.collision = True
            return -self.G * np.ones(self.n_agent)
        # headwayReward = - 标准头距差的平方
        h_rewards = -(self.hs_cur - self.h_star) ** 2
        # speedReward = - 车速reward因子 * 标准车速差的平方
        v_rewards = -self.a * (self.vs_cur - self.v_star) ** 2
        # accReward = - 加速reward因子 * 加速度的平方
        u_rewards = -self.b * (self.us_cur) ** 2
        if self.train_mode:
            # 训练过程可以给予碰撞奖励
            c_rewards = -COLLISION_WT * (np.minimum(self.hs_cur - COLLISION_HEADWAY, 0)) ** 2
        else:
            c_rewards = 0
        rewards = h_rewards + v_rewards + u_rewards + c_rewards
       # rewards = v_rewards
        return rewards

    def state2Reward(self, state):
        """
            state of shape [b, 8, 5] to reward of shape [b, 8]
        """
        device = state.device
        state = state.detach().cpu()
        v_state = state[:, :, 0]
        h_state = state[:, :, 3]
        u_state = state[:, :, 4]

        v = v_state*self.v_star + self.v_star
        u = u_state*self.u_max
        h = h_state * self.h_star + self.h_star
        v0 = [self.v0s[self.t]]*v.shape[0]
        v0 = torch.tensor(v0).unsqueeze(1)
        # [b, 1]
        vlead = torch.cat([v0, v[:, :-1]], dim=1)
        h = h - self.dt*(vlead - v)

        h_rewards = -(h - self.h_star) ** 2
        v_rewards = -self.a * (v - self.v_star) ** 2
        u_rewards = -self.b * (u) ** 2

        collision =  torch.min(h, dim=1, keepdim=True)[0] < self.h_min
        collision = collision.float()
        rewards = h_rewards + v_rewards + u_rewards
        #rewards = v_rewards
        rewards = (1-collision)*rewards + (collision *-self.G * torch.ones(self.n_agent))

        rewards = rewards.to(device)
        collision = collision.to(device)
        return rewards, collision.expand(*rewards.shape) # ignore done because of max-step

    # NOTE: state space
    def _get_veh_state(self, i_veh):
        '''
        Obtain state of vehicle i_veh
        Args:
            i_veh: index of the current vehicle
        Return:
            state_list: [v_state, vdiff_state, vhdiff_state, h_state, u_state]
        '''
        # v_lead: 前车车速

        v_lead = self.vs_cur[i_veh-1] if i_veh else self.v0s[self.t]

        # v_state: (当前车速-标准车速)/标准车速
        v_state = (self.vs_cur[i_veh] - self.v_star) / self.v_star
        # vdiff_state: -2clip2(前车车速-当前车速)/标准车速差
        vdiff_state = np.clip((v_lead - self.vs_cur[i_veh]) / VDIFF, -2, 2)
        # vh: 目标速度
        vh = self.ovm.get_vh(self.hs_cur[i_veh])
        # vhdiff_state: -2clip2(目标车速-当前车速)/标准车速差
        vhdiff_state = np.clip((vh - self.vs_cur[i_veh]) / VDIFF, -2, 2)
        # h_state: (当前headway + 与前车速度差 * 时间 - 标准headway) / 标准headway
        h_state = (self.hs_cur[i_veh] + (v_lead-self.vs_cur[i_veh])*self.dt - self.h_star) / self.h_star
        # v_state = np.clip((self.vs_cur[i_veh] - self.v_star) / self.v_norm, -2, 2)
        # h_state = np.clip((self.hs_cur[i_veh] - self.h_star) / self.h_norm, -2, 2)
        # u_state: 当前加速度/最大加速度
        u_state = self.us_cur[i_veh] / self.u_max

        if RECORD_FLAG == True:
            print(f'INDEX {i_veh}: current headway {self.hs_cur[i_veh]}, last acceleration {self.us_cur[i_veh]}, current velocity {self.vs_cur[i_veh]}, v_lead {v_lead}, vh {vh}')
            if RECORD_FLAG == True:
                subfolders = ["headway", "acceleration", "velocity"]
                for subfolder in subfolders:
                    (folder_path / subfolder).mkdir(parents=True, exist_ok=True)
                    headway_dir = os.path.join(folder_path, subfolder)
                    os.makedirs(headway_dir, exist_ok=True)
                    # file_path = os.path.join(headway_dir, f"{i_veh+1}.txt")
                    if subfolder == "headway":
                        with open(f"{folder_path}/headway/{i_veh+1}.txt", "a") as file:
                            file.write(f"{self.hs_cur[i_veh]}\n")
                    elif subfolder =="acceleration":
                        with open(f"{folder_path}/acceleration/{i_veh+1}.txt", "a") as file:
                            file.write(f"{self.us_cur[i_veh]}\n")
                    elif subfolder == "velocity":
                        with open(f"{folder_path}/velocity/{i_veh+1}.txt", "a") as file:
                            file.write(f"{self.vs_cur[i_veh]}\n")

        return np.array([v_state, vdiff_state, vhdiff_state, h_state, u_state])

    # NOTE: state for LLM +
    def transfer_state_llm(self, state_tensor):
        state_list = state_tensor.tolist()
        state_list_llm = []
        # NOTE: may wrong -------------------------
        # for v_index,state in enumerate(state_list):
        #     v_state = state[0]
        #     # v_speed, v_star
        #     v_speed1 = v_state*self.v_star + self.v_star
        #     v_speed2 = self.vs_cur[v_index]
        #     if v_speed1 - v_speed2 > 0.1:
        #         import pdb
        #         pdb.set_trace()
        #     else:
        #         v_speed = v_speed2
        #         # v_lead
        #         v_lead = self.vs_cur[v_index-1] if v_index else self.v0s[self.t]
        #         # vh
        #         v_vh = self.ovm.get_vh(self.hs_cur[v_index])
        #         # h_headway
        #         h_state = self.hs_cur[v_index] + (v_lead-self.vs_cur[v_index])*self.dt
        #         # u_state
        #         u_state = self.us_cur[v_index]
        #         state_llm = [v_lead, v_speed, self.v_star, v_vh, h_state, self.h_star, u_state]
        #     state_list_llm.append(state_llm)
        for v_index,state in enumerate(state_list):
            v_lead = self.vs_cur[v_index-1] if v_index else self.v0s[self.t]
            v_speed = self.vs_cur[v_index]
            v_vh = self.ovm.get_vh(self.hs_cur[v_index])
            state_llm = [v_lead, v_speed, self.v_star, v_vh, self.hs_cur[v_index], self.h_star, self.us_cur[v_index]]
            state_list_llm.append(state_llm)
        return state_list_llm



    # NOTE: state space
    # State for the LLM without standardization.
    def _get_veh_llm_state(self, i_veh):
        '''
        Obtain state of vehicle i_veh
        Args:
            i_veh: index of the current vehicle
        Return:
            state_list: [v_state, vdiff_state, vhdiff_state, h_state, u_state]
        '''
        # lead vehicle speed
        v_lead = self.vs_cur[i_veh-1] if i_veh else self.v0s[self.t]
        # current vehicle speed
        v_speed = self.vs_cur[i_veh]
        # objective catchup speed
        v_obj = self.v_star
        #
        v_vh = self.ovm.get_vh(self.hs_cur[i_veh])
        h_state = self.hs_cur[i_veh] + (v_lead-self.vs_cur[i_veh])*self.dt
        h_obj = self.h_star
        u_state = self.us_cur[i_veh]
        return np.array([v_lead, v_speed, v_obj, v_vh, h_state, h_obj, u_state])

    def _get_state(self):
        state = []
        for i in range(self.n_agent):
            cur_state = [self._get_veh_state(i)]
            if self.agent.startswith('ia2c'):
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    cur_state.append(self._get_veh_state(j))
            if self.agent == 'ia2c_fp':
                # finger prints must be attached at the end of the state array
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    cur_state.append(self.fp[j])
            state.append(np.concatenate(cur_state))
        return state

    # State for the LLM without normalization.
    def _get_llm_state(self):
        state = []
        for i in range(self.n_agent):
            cur_state = [self._get_veh_llm_state(i)]
            if self.agent.startswith('ia2c'):
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    cur_state.append(self._get_veh_llm_state(j))
            if self.agent == 'ia2c_fp':
                # finger prints must be attached at the end of the state array
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    cur_state.append(self.fp[j])
            state.append(np.concatenate(cur_state))
        return state

    def _log_control_data(self, action, global_reward):
        action_r = ','.join(['%d' % a for a in action])
        cur_control = {'episode': self.cur_episode,
                       'time_sec': self.t * self.dt,
                       'step': self.t,
                       'action': action_r,
                       'reward': global_reward}
        self.control_data.append(cur_control)

    def _log_traffic_data(self):
        hs = np.array(self.hs)
        vs = np.array(self.vs)
        us = np.array(self.us)
        df = pd.DataFrame()
        df['episode'] = np.ones(len(hs)) * self.cur_episode
        df['time_sec'] = np.arange(len(hs)) * self.dt
        df['reward'] = np.array(self.rewards)
        df['lead_headway_m'] = hs[:, 0]
        df['avg_headway_m'] = np.mean(hs[:, 1:], axis=1)
        df['std_headway_m'] = np.std(hs[:, 1:], axis=1)
        df['avg_speed_mps'] = np.mean(vs, axis=1)
        df['std_speed_mps'] = np.std(vs, axis=1)
        df['avg_accel_mps2'] = np.mean(us, axis=1)
        df['std_accel_mps2'] = np.std(us, axis=1)
        for i in range(self.n_agent):
            df['headway_%d_m' % (i+1)] = hs[:, i]
            df['velocity_%d_mps' % (i+1)] = vs[:, i]
            df['accel_%d_mps2' % (i+1)] = us[:, i]
        self.traffic_data.append(df)

    def collect_tripinfo(self):
        return

    def init_data(self, is_record, record_stats, output_path):
        self.is_record = is_record
        self.output_path = output_path
        if self.is_record:
            self.control_data = []
            self.traffic_data = []

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds

    def get_neighbor_action(self, action):
        naction = []
        for i in range(self.n_agent):
            naction.append(action[self.neighbor_mask[i] == 1])
        return naction

    def output_data(self):
        if not self.is_record:
            logging.error('Env: no record to output!')
        control_data = pd.DataFrame(self.control_data)
        control_data.to_csv(self.output_path + ('%s_%s_control.csv' % (self.name, self.agent)))
        traffic_data = pd.concat(self.traffic_data)
        traffic_data.to_csv(self.output_path + ('%s_%s_traffic.csv' % (self.name, self.agent)))
        # self.plot_data(df, path)

    # Moved to python notebook
    # def plot_data(self, df, path):
    #     fig = plt.figure(figsize=(10, 8))
    #     plt.subplot(2, 1, 1)
    #     for i in [0, 2, 5, 7]:
    #         plt.plot(df.time.values, df['headway_%d' % (i+1)].values, linewidth=3,
    #                  label='veh #%d' % (i+1))
    #     plt.legend(fontsize=20, loc='best')
    #     plt.grid(True, which='both')
    #     plt.yticks(fontsize=20)
    #     plt.xticks(fontsize=20)
    #     plt.ylabel('Headway [m]', fontsize=20)
    #     plt.subplot(2, 1, 2)
    #     for i in [0, 2, 5, 7]:
    #         plt.plot(df.time.values, df['velocity_%d' % (i+1)].values, linewidth=3,
    #                  label='veh #%d' % (i+1))
    #     # plt.legend(fontsize=15, loc='best')
    #     plt.grid(True, which='both')
    #     plt.yticks(fontsize=20)
    #     plt.xticks(fontsize=20)
    #     plt.ylabel('Velocity [m/s]', fontsize=20)
    #     plt.xlabel('Time [s]', fontsize=20)
    #     fig.tight_layout()
    #     plt.savefig(path + 'env_plot.pdf')
    #     plt.close()

    def reset(self, gui=False, test_ind=-1):
        self.cur_episode += 1
        # np.random.seed(self.seed)
        if (self.train_mode):
            seed = self.seed
        elif (test_ind < 0):
            seed = self.seed-1
        else:
            seed = self.test_seeds[test_ind]
        np.random.seed(seed)
        self.seed += 1
        self._init_common()
        if self.name.startswith('catchup'):
            self._init_catchup()
        elif self.name.startswith('slowdown'):
            self._init_slowdown()
        self.collision = False
        self.hs_cur = self.hs[0]
        # TODO:
        # self.hs_cur = self.hs[0]+180
        self.vs_cur = self.vs[0]
        self.us_cur = np.zeros(self.n_agent)
        self.fp = np.ones((self.n_agent, self.n_a)) / self.n_a
        self.us = [self.us_cur]
        self.rewards = [0]
        return self._get_state()

    def step(self, action):
        # if collision happens, return -G for all the remaining steps
        if self.collision:
            reward = -self.G * np.ones(self.n_agent)
        else:
            rl_params = [self.a_map[a] for a in action]
            hs_next = []
            vs_next = []
            self.us_cur = []
            # update speed
            for i in range(self.n_agent):
                # h_g = rl_hgs[i]
                # u = self._get_accel(i, h_g)
                cur_alpha, cur_beta = rl_params[i]
                u = self._get_accel(i, cur_alpha, cur_beta)
                # apply v, u constraints
                v_next, u_const = self._constrain_speed(self.vs_cur[i], u)
                self.us_cur.append(u_const)
                if RECORD_FLAG == True:
                    print(f'u: {u}, cur_alpha: {cur_alpha}, cur_beta: {cur_beta}, u_const: {u_const}')

                vs_next.append(v_next)
            # update headway

            for i in range(self.n_agent):
                if i == 0:
                    v_lead = self.v0s[self.t]
                    v_lead_next = self.v0s[self.t+1]
                else:
                    v_lead = self.vs_cur[i-1]
                    v_lead_next = vs_next[i-1]
                v = self.vs_cur[i]
                v_next = vs_next[i]
                hs_next.append(self.hs_cur[i] + 0.5*self.dt*(v_lead+v_lead_next-v-v_next))

                # print(f'hs_next: {hs_next}, v_lead: {v_lead}, v_lead_next: {v_lead_next}, v: {v}, v_next: {v_next}')
                # a_lead = (v_lead_next - v_lead) / self.dt
                # s_lead = v_lead * self.dt + 0.5 * a_lead * self.dt ** 2
                # s_self = v * self.dt + 0.5 * u_const * self.dt ** 2
                # next_hs = self.hs_cur[i] + (s_lead - s_self)
                # print(f's_lead: {s_lead}, s_self: {s_self}, next_hs: {next_hs}')
            self.hs_cur = np.array(hs_next)
            self.vs_cur = np.array(vs_next)
            self.us_cur = np.array(self.us_cur)
            reward = self._get_reward()
        if RECORD_FLAG == True:
            print(f'reward: {reward}')
        self.hs.append(self.hs_cur)
        self.vs.append(self.vs_cur)
        self.us.append(self.us_cur)
        self.t += 1
        global_reward = np.sum(reward)
        self.rewards.append(global_reward)
        done = False
        if self.collision and not self.t % self.batch_size:
            done = True
        if self.t == self.T:
            done = True
        if self.is_record:
            self._log_control_data(action, global_reward)
        if done and (self.is_record):
            self._log_traffic_data()
        return self._get_state(), reward, done, global_reward

    # NOTE: LLM_step
    def llm_step(self, action):
        # if collision happens, return -G for all the remaining steps
        if self.collision:
            reward = -self.G * np.ones(self.n_agent)
        else:
            # rl_params = [self.a_map[a] for a in action] # NOTE: delete
            hs_next = []
            vs_next = []
            self.us_cur = []
            # update speed
            for i in range(self.n_agent):
                # h_g = rl_hgs[i]
                # u = self._get_accel(i, h_g)
                # cur_alpha, cur_beta = rl_params[i] # NOTE: delete
                # u = self._get_accel(i, cur_alpha, cur_beta) # NOTE: delete
                u = action[i]
                # apply v, u constraints
                v_next, u_const = self._constrain_speed(self.vs_cur[i], u)
                self.us_cur.append(u_const)
                vs_next.append(v_next)
            # update headway
            for i in range(self.n_agent):
                if i == 0:
                    v_lead = self.v0s[self.t]
                    v_lead_next = self.v0s[self.t+1]
                else:
                    v_lead = self.vs_cur[i-1]
                    v_lead_next = vs_next[i-1]
                v = self.vs_cur[i]
                v_next = vs_next[i]
                hs_next.append(self.hs_cur[i] + 0.5*self.dt*(v_lead+v_lead_next-v-v_next))
            self.hs_cur = np.array(hs_next)
            self.vs_cur = np.array(vs_next)
            self.us_cur = np.array(self.us_cur)
            reward = self._get_reward()
        self.hs.append(self.hs_cur)
        self.vs.append(self.vs_cur)
        self.us.append(self.us_cur)
        self.t += 1
        global_reward = np.sum(reward)
        self.rewards.append(global_reward)
        done = False

        # if self.collision and not self.t % (self.batch_size*2):
        if self.collision:
            # import pdb; pdb.set_trace()
            done = True
        if self.t == self.T:
            # import pdb; pdb.set_trace()
            done = True
        if self.is_record:
            self._log_control_data(action, global_reward)
        if done and (self.is_record):
            self._log_traffic_data()
        return self._get_state(), reward, done, global_reward


    def get_fingerprint(self):
        return self.fp

    def update_fingerprint(self, fp):
        self.fp = fp

    def terminate(self):
        return

    # NOTE: action sapce
    def _init_space(self):
        self.neighbor_mask = np.zeros((self.n_agent, self.n_agent)).astype(int)
        self.distance_mask = np.zeros((self.n_agent, self.n_agent)).astype(int)
        cur_distance = list(range(self.n_agent))
        for i in range(self.n_agent):
            self.distance_mask[i] = cur_distance
            cur_distance = [i+1] + cur_distance[:-1]
            if i >= 1:
                self.neighbor_mask[i,i-1] = 1
            # if i >= 2:
            #     self.neighbor_mask[i,i-2] = 1
            if i <= self.n_agent-2:
                self.neighbor_mask[i,i+1] = 1
            # if i <= self.n_agent-3:
            #     self.neighbor_mask[i,i+2] = 1
        # 5 levels of high level control: conservative -> aggressive

        #self.neighbor_mask = np.identity(self.n_agent)

        self.n_a_ls = [4] * self.n_agent
        self.n_a = 4
        # a_interval = (self.h_g - self.h_s) / ((self.n_a+1)*0.5)
        # disable OVM h_g as action!
        # self.a_map = np.arange(-10, 11, 5) + self.h_g
        # a_map = (alpha, beta)
        self.a_map = [(0,0), (0.5,0), (0,0.5), (0.5,0.5)]
        logging.info('action to h_go map:\n %r' % self.a_map)
        self.n_s_ls = []
        for i in range(self.n_agent):
            if self.agent.startswith('ma2c'):
                num_n = 1
            else:
                num_n = 1 + np.sum(self.neighbor_mask[i])
            self.n_s_ls.append(num_n * 5)

    def _init_catchup(self):
        # first vehicle has long headway (4x) and remaining vehicles have random
        # headway (1 x~1.5x)
        #self.hs = [(1+0.5*np.random.rand(self.n_agent)) * self.h_star]
        self.hs = [np.ones(self.n_agent) * self.h_star]
        if not self.seed:
            self.hs[0][0] = self.h_star*2
        else:
            # s = [0, -1, -0.5, 0.5, 1]
            self.hs[0][0] = self.h_star*(1.5+np.random.rand())
            # self.hs[0][0] = self.h_star*(4+s[self.seed])
        # all vehicles have v_star initially
        self.vs = [np.ones(self.n_agent) * self.v_star]
        # leading vehicle (before platoon) is driving at v_star
        self.v0s = np.ones(self.T+1) * self.v_star



    def _init_common(self):
        self.alpha = 0.5
        self.beta = 0.5
        self.t = 0

    def _init_slowdown(self):
        # all vehicles have random headway (1x~1.5x)
        # self.hs = [(1+0.5*np.random.rand(self.n_agent)) * self.h_star]
        self.hs = [np.ones(self.n_agent) * self.h_star]
        # all vehicles have 2v_star initially
        if not self.seed:
            self.vs = [np.ones(self.n_agent) * 2*self.v_star]
        else:
            self.vs = [np.ones(self.n_agent) * self.v_star*(1.5+np.random.rand())]
        # leading vehicle is decelerating from 2v_star to v_star with 0.02*u_min
        self.v0s = np.ones(self.T+1) * self.v_star
        v0s_decel = np.linspace(self.vs[0][0], self.v_star, 30)
        # v0s_decel = np.linspace(self.vs[0][0], self.v_star, 300)
        self.v0s[:len(v0s_decel)] = v0s_decel

    def _load_config(self, config):
        # self.dt = config.getfloat('control_interval_sec')
        self.dt = config.getfloat('control_interval_sec')
        self.T = int(config.getint('episode_length_sec') / self.dt)
        self.batch_size = config.getint('batch_size')
        self.h_min = config.getfloat('headway_min')
        self.h_star = config.getfloat('headway_target')
        self.h_norm = config.getfloat('norm_headway')
        self.h_s = config.getfloat('headway_st')
        self.h_g = config.getfloat('headway_go')
        self.v_max = config.getfloat('speed_max')
        self.v_star = config.getfloat('speed_target')
        self.v_norm = config.getfloat('norm_speed')
        self.u_min = config.getfloat('accel_min')
        self.u_max = config.getfloat('accel_max')
        self.name = config.get('scenario').split('_')[1]
        self.a = config.getfloat('reward_v')
        self.b = config.getfloat('reward_u')
        self.G = config.getfloat('collision_penalty')
        self.n_agent = config.getint('n_vehicle')
        self.agent = config.get('agent')
        self.coop_gamma = config.getfloat('coop_gamma')
        self.seed = config.getint('seed')
        test_seeds = [int(s) for s in config.get('test_seeds').split(',')]
        self.init_test_seeds(test_seeds)


class OVMCarFollowing:
    '''
    The OVM controller for vehicle ACC
    Attributes:
        h_st (float): stop headway # min headway
        h_go (float): full-speed headway # 全速headway
        v_max (float): max speed 车辆速度上限
    '''
    def __init__(self, h_st, h_go, v_max):
        """Initialization."""
        self.h_st = h_st
        self.h_go = h_go
        self.v_max = v_max

    def get_vh(self, h, h_go=-1):
        '''
        Calculate objective speed
        Args:
            h: headway
            h_go: full-speed headway 当前车与前车的距离大于这个值，后车可以以最大速度加速行驶
        '''
        if h_go < 0:
            h_go = self.h_go
        if h <= self.h_st:
            # headway < stop threshold, v=0
            return 0
        elif self.h_st < h < h_go:
            # stop headway < headway < full speed headway, 根据当前的位置返回合适的v, [0, self.v_max](h-self.h_st) / (1 - np.cos(np.pi * (h-self.h_st) / (h_go-self.h_st)))\in[0,2]
            return self.v_max / 2 * (1 - np.cos(np.pi * (h-self.h_st) / (h_go-self.h_st)))
            # vh = self.v_max * ((d-h_st) / (h_go-h_st))
        else:
            return self.v_max

    def get_accel(self, v, v_lead, h, alpha, beta, h_go=-1):
        """
        Get target acceleration using OVM controller.
        Args:
            v (float): current vehicle speed
            v_lead (float): leading vehicle speed
            h (float): current headway
            alpha, beta (float): human parameters
        Returns:
            accel (float): target acceleration
        """
        vh = self.get_vh(h, h_go=h_go)
        # alpha is applied to both headway based V and leading speed based V.
        # 目标速度差
        return alpha*(vh-v) + beta*(v_lead-v)


if __name__ == '__main__':
    count = 0
    output_path = PROJECT_ROOT / 'algorithms/envs/NCS/test'
    config_path = PROJECT_ROOT / 'algorithms/envs/NCS/config/config_ma2c_nc_slowdown.ini'
    config = configparser.ConfigParser()
    config.read(str(config_path))
    env = CACCEnv(config['ENV_CONFIG'])
    env.init_data(True, False, str(output_path))
    ob = env.reset()

    while True:
        current_action = []
        current_state = env.transfer_state_llm(state_tensor=np.array(ob))
        for agent_index in range(8):
            agent_state = current_state[agent_index]
            my_headway = agent_state[4]
            my_velocity = agent_state[1]
            ahead_velocity = agent_state[0]
            weight_flag = float(config["LLM_CONFIG"]["W_FLAG"])
            v_flag = int(config["LLM_CONFIG"]["VELOCITY_FLAG"])
            opt_acc, w_headway, headway_optimal_acc, w_velocity, velocity_optimal_acc = optimal_acc(ahead_velocity, my_headway, my_velocity, flag = weight_flag, v_flag = v_flag, dt=0.5, task_flag=TASK_FLAG)
            if opt_acc > 2.5:
                opt_acc = 2.5
            if opt_acc < -2.5:
                opt_acc = -2.5
            current_action.append(opt_acc)
        count += 1
        # ob, _, done, _ = env.step([1]*(env.n_agent))
        ob, reward, done, global_reward = env.llm_step(current_action)
        # import pdb; pdb.set_trace()
        print(f"current_count{count}")
        # if count % 10 ==0:
        # if count > 20 and count < 30:
        #     import pdb; pdb.set_trace()
        if done:
            break
    env.output_data()
