import cv2
import numpy as np
import l2arcadekit.l2agames as l2agames
from utils.utils_env_rl import l2arcade_noop_reset

#####################################################################################################
##########           Functions and classes for wrapper of L2M Arcade environment           ##########
#####################################################################################################
def remove_unnecessary_param(orig_env_params):
    temp_params = dict(orig_env_params)
    if 'game_name' in temp_params.keys():
        _ = temp_params.pop('game_name')
    return temp_params

class L2M_Arcade_Wrapper():
    def __init__(self, game_name, env_params=None, max_steps=100, noop_reset_max_steps=-1, visualize=False, visualize_window_cnt=0):
        self.game_name = game_name
        self.max_steps = max_steps
        self.noop_reset = (noop_reset_max_steps > 0)
        self.noop_reset_max_steps = noop_reset_max_steps
        self.visualize = visualize
        self.visualize_window_cnt = visualize_window_cnt

        if self.game_name.lower() == 'pong':
            self.reward_scale = 1.0
        else:
            self.reward_scale = 1.0

        self.game = getattr(l2agames, game_name)()
        self.observation_space = self.game.observation_space
        self.action_space = self.game.action_space
        if env_params is None:
            env_params = self.game._external_params.get_defaults()
        else:
            env_params = remove_unnecessary_param(env_params)
        self.game.validate(**env_params)
        self.env_params = env_params

        self.obs = self.game.reset(**self.env_params)

        if self.visualize:
            cv2.namedWindow("env-%d"%(self.visualize_window_cnt))

    def close(self):
        print("Close environment!")
        if self.visualize:
            cv2.destroyWindow("env-%d"%(self.visualize_window_cnt))
        del self.game

    def reset(self):
        self.step_cnt = 0
        if self.noop_reset:
            self.obs = l2arcade_noop_reset(self.game, self.noop_reset_max_steps, self.env_params)
        else:
            self.obs = self.game.reset(**self.env_params)
        return self.obs

    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        self.obs = obs
        self.step_cnt += 1
        if self.step_cnt >= self.max_steps:
            done = True
        return obs, reward*self.reward_scale, done, info

    def render(self, time_to_wait=20):
        if self.visualize:
            cv2.imshow("env-%d"%(self.visualize_window_cnt), self.obs)
            cv2.waitKey(time_to_wait)


class L2M_Arcade_Wrapper_A2C():
    def __init__(self, game_name, num_game, env_params=None, max_steps=100, noop_reset_max_steps=-1, visualize=False, visual_window_start_index=0):
        self.num_game = num_game
        self.visualize = visualize
        self.visual_window_start_index = visual_window_start_index
        self.game = [L2M_Arcade_Wrapper(game_name, env_params, max_steps, noop_reset_max_steps, visualize, visual_window_start_index+i) for i in range(num_game)]

        self.obs = self.reset()
        self.env_params = self.game[0].env_params
        self.observation_space = self.game[0].observation_space
        self.obs_shape = [self.num_game] + list(self.observation_space.shape)
        self.action_space = self.game[0].action_space

    def close(self):
        print("Close A2C environments!")
        for game in self.game:
            game.close()
        del self.game

    def update_steps(self):
        self.step_cnts = [game.step_cnt for game in self.game]

    def reset(self, game_index=-1):
        if game_index >= 0 and game_index < self.num_game:
            tmp_obs = self.game[game_index].reset()
            self.obs[game_index,:,:,:] = tmp_obs
        elif game_index < 0:
            obs_stack = []
            for game in self.game:
                tmp_obs = game.reset()
                obs_stack.append(tmp_obs)
            self.obs = np.array(obs_stack)
        else:
            print("\n\nGiven index of game to reset is wrong!\n\n")
            raise ValueError
        self.update_steps()
        return self.obs

    def step(self, actions):
        assert (len(actions) == self.num_game), "Given action doesn't match the number of available environments!"
        obs_stack, reward_stack, done_stack, infos = [], [], [], []
        for game, act in zip(self.game, actions):
            obs, reward, done, info = game.step(act)
            if done:
                obs = game.reset()
            obs_stack.append(obs)
            reward_stack.append(reward)
            done_stack.append(done)
            infos.append(info)
        self.obs, rewards, dones = np.array(obs_stack), np.array(reward_stack), np.array(done_stack)
        return self.obs, rewards, dones, infos

    def render(self, time_to_wait=20):
        if self.visualize:
            for game in self.game:
                game.render(time_to_wait=1)
            cv2.waitKey((time_to_wait))


#####################################################################################################
##########           Functions to store hyper-parameter of environment and model           ##########
#####################################################################################################
def reformat_for_text_writer(value_to_write):
    if type(value_to_write) == list:
        tmp = "["
        for i, (v) in enumerate(value_to_write):
            tmp += str(v)
            if i < len(value_to_write)-1:
                tmp += ","
            else:
                tmp += "]"
    else:
        tmp = str(value_to_write)
    return tmp

def save_training_details(model_param, rl_param, logging_path, train_mode=-1, pd_teacher_param_files=None):
    with open(logging_path + '/train_details.txt', 'w') as fobj:
        fobj.write("[Model Parameters]\n")
        key_list = list(model_param.keys())
        key_list.sort()
        for key in key_list:
            fobj.write(key)
            fobj.write('\t\t')
            fobj.write(reformat_for_text_writer(model_param[key]))
            fobj.write('\n')

        fobj.write("\n[RL Training Parameters]\n")
        key_list = list(rl_param.keys())
        key_list.sort()
        for key in key_list:
            if not ('env' in key):
                fobj.write(key)
                fobj.write('\t\t')
                fobj.write(reformat_for_text_writer(rl_param[key]))
                fobj.write('\n')

        if train_mode == 0:
            fobj.write('training mode\t\tchanging env at the termination of an episode\n')
        elif train_mode == 1:
            fobj.write('training mode\t\tchanging env at every mini-batch\n')

        if pd_teacher_param_files is not None:
            fobj.write('policy distillation > teacher param files\t\t')
            fobj.write(reformat_for_text_writer(pd_teacher_param_files))
            fobj.write('\n')

        fobj.write("\nEnvironments\t\t")
        fobj.write(reformat_for_text_writer(rl_param['env']))
        fobj.write('\n')
        key_list = list(rl_param.keys())
        key_list.sort()
        for key in key_list:
            if 'env_' in key:
                fobj.write(key)
                fobj.write('\t\t')
                fobj.write(reformat_for_text_writer(rl_param[key]))
                fobj.write('\n')