from os import getcwd, listdir, mkdir
import csv
import random

import l2arcadekit.l2agames as l2agames


def l2arcade_noop_reset(env, num_max_steps, task_params):
    obs = env.reset(**task_params)
    num_steps = random.randint(0, num_max_steps)
    for cnt in range(num_steps):
        obs, _, done, _ = env.step(0)
        if done:
            obs = env.reset(**task_params)
    return obs


###################################################################################################
########## Functions of generating and/or loading parameters of multiple RL environments ##########
###################################################################################################
def write_params_to_csv(list_of_task_params, csv_file_name):
    with open(csv_file_name, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for task_cnt, (task_params) in enumerate(list_of_task_params):
            csv_writer.writerow(['<Task-%d>'%(task_cnt)])
            for key, val in task_params.items():
                csv_writer.writerow([key, val])
        print("\tSaved new set of parameters for tasks at %s\n"%(csv_file_name))

def convert_data_type(data_in_string):
    if (data_in_string[0] == '[') and (data_in_string[-1] == ']'):
        result, tmps = [], data_in_string[1:-1].split(',')
        for tmp in tmps:
            result.append(convert_data_type(tmp))
    elif '.' in data_in_string:
        result = float(data_in_string)
    else:
        try:
            result = int(data_in_string)
        except:
            result = str(data_in_string)
    return result

def load_params_from_csv(csv_file_name):
    list_of_task_params, task_cnt = [], 0
    with open(csv_file_name, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if 'Task' in row[0]:
                if task_cnt > 0:
                    list_of_task_params.append(task_params)
                task_params, task_cnt = {}, task_cnt + 1
            elif len(row) == 2:
                task_params[row[0]] = convert_data_type(row[1])
            else:
                print("There is a problem with the format of parameter file!")
                raise ValueError
        if 'task_params' in globals() or 'task_params' in locals():
            list_of_task_params.append(task_params)
    return list_of_task_params


###################################################################################################
def task_generator(env_type, num_envs, task_param_file_name, max_prop_cnt_to_change=0, change_rotation=False):
    if 'RL' not in listdir(getcwd()+'/Data'):
        mkdir(getcwd()+'/Data/RL')
    if 'APL_testbed' not in listdir(getcwd()+'/Data/RL'):
        mkdir(getcwd()+'/Data/RL/APL_testbed')

    if 'pong' in env_type.lower():
        game_name = 'Pong'
    elif 'breakout' in env_type.lower():
        game_name = 'Breakout'
    elif 'ballvpaddle' in env_type.lower():
        game_name = 'BallvPaddle'
    elif 'freeway' in env_type.lower():
        game_name = 'Freeway'
    else:
        print("Not valid game type is specified!")
        raise ValueError

    ## Load or Generate parameters of tasks
    param_file_name_w_path = getcwd()+'/Data/RL/APL_testbed/'+task_param_file_name
    if task_param_file_name in listdir(getcwd()+'/Data/RL/APL_testbed/'):
        tasks_params = load_params_from_csv(param_file_name_w_path)
        print("Tasks' parameters are successfully loaded!")
    else:
        tmp_game = getattr(l2agames, game_name)
        tasks_params, default_game_params = [], tmp_game._external_params.get_defaults()
        num_properties = len(default_game_params.keys())
        if 'soup' in env_type.lower():
            for env_cnt in range(num_envs):
                new_params, temp = dict(default_game_params), tmp_game._external_params.sample()
                new_params['game_name'] = game_name
                new_params['img/noise'] = temp['img/noise']
                new_params['bg_color'] = temp['bg_color']
                if change_rotation:
                    new_params['img/rotation'] = temp['img/rotation']
                tasks_params.append(new_params)
        elif 'diverse' in env_type.lower():
            for env_cnt in range(num_envs):
                new_params, temp = dict(default_game_params), tmp_game._external_params.sample()
                list_of_keyas_to_modify = list(default_game_params.keys())
                if not change_rotation:
                    list_of_keyas_to_modify.remove('img/rotation')
                for key_to_change in random.sample(list_of_keyas_to_modify, random.randint(1, len(list_of_keyas_to_modify)) if max_prop_cnt_to_change < 1 or max_prop_cnt_to_change > len(list_of_keyas_to_modify) else random.randint(1, max_prop_cnt_to_change)):
                    new_params[key_to_change] = temp[key_to_change]
                new_params['game_name'] = game_name
                tasks_params.append(new_params)

        print("Tasks' parameters are successfully generated!")
        write_params_to_csv(tasks_params, param_file_name_w_path)
    return tasks_params



def model_setup(env_type, model_type, test_type=0, skip_connect_test_type=0, highway_connect_test_type=0, boltz_exp=False):
    model_hyperpara = {}

    model_hyperpara['padding_type'] = 'SAME'
    #model_hyperpara['padding_type'] = 'VALID'

    model_hyperpara['max_pooling'] = True
    model_hyperpara['dropout'] = True
    model_hyperpara['skip_connect'] = []
    model_hyperpara['highway_connect'] = highway_connect_test_type
    model_hyperpara['boltzmann_exploration'] = boltz_exp

    #### L2 Arcade tasks
    if 'soup' in env_type.lower():
        model_hyperpara['batch_size'] = 32
        if 'ffnn' in model_type.lower():
            model_hyperpara['hidden_layer'] = [1024, 256, 128]
        elif 'random' not in model_type.lower():
            ## CNN-FFNN case
            model_hyperpara['hidden_layer'] = [128, 32]
            '''
            model_hyperpara['kernel_sizes'] = [5, 5, 5, 5, 3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [32, 64, 128, 256]
            model_hyperpara['pooling_size'] = [2, 2, 2, 2, 2, 2, 2, 2]
            '''
            model_hyperpara['kernel_sizes'] = [5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [32, 64, 128, 128, 128, 128]
            model_hyperpara['pooling_size'] = [2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2]

        if skip_connect_test_type > 0:
            for _ in range(10):
                print("\tSkip connection isn't set up properly!!")
            #model_hyperpara['skip_connect']

    elif 'diverse' in env_type.lower():
        model_hyperpara['batch_size'] = 32
        if 'ffnn' in model_type.lower():
            model_hyperpara['hidden_layer'] = [1024, 256, 128]
        elif 'random' not in model_type.lower():
            ## CNN-FFNN case
            model_hyperpara['hidden_layer'] = [128, 32]
            model_hyperpara['kernel_sizes'] = [5, 5, 5, 5, 3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [32, 64, 128, 256]
            model_hyperpara['pooling_size'] = [2, 2, 2, 2, 2, 2, 2, 2]

        if skip_connect_test_type > 0:
            for _ in range(10):
                print("\tSkip connection isn't set up properly!!")
            #model_hyperpara['skip_connect']

    if model_type.lower() == 'random':
        model_hyperpara['model_type'] = 'random_agent'
    elif model_type.lower() == 'debug_stl':
        model_hyperpara['model_type'] = 'conv_qnet'
    elif model_type.lower() == 'debug_stl_ffnn':
        model_hyperpara['model_type'] = 'ffnn_qnet'
    elif model_type.lower() == 'stl':
        model_hyperpara['model_type'] = 'mt_several_conv_net'
    elif model_type.lower() == 'hps':
        model_hyperpara['model_type'] = 'mt_hard_param_sharing_conv_net'

    elif model_type.lower() == 'dfcnn':
        model_hyperpara['model_type'] = 'mt_hybrid_dfcnn_net'
        model_hyperpara['dfconv_reg_scale'] = [0.0, 0.0, 0.0, 0.0]
        model_hyperpara['conv_skip_connect'] = []
        if len(model_hyperpara['kernel_sizes']) == 8:
            model_hyperpara['dfconv_KB_sizes'] = [3, 12, 3, 32, 2, 64, 2, 72]
            model_hyperpara['dfconv_TS_sizes'] = [3, 24, 3, 64, 3, 128, 3, 144]
            model_hyperpara['dfconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
        elif len(model_hyperpara['kernel_sizes']) == 12:
            model_hyperpara['dfconv_KB_sizes'] = [3, 16, 3, 32, 2, 72, 2, 96, 2, 96, 2, 96]
            model_hyperpara['dfconv_TS_sizes'] = [3, 24, 3, 64, 3, 128, 3, 160, 2, 160, 2, 160]
            model_hyperpara['dfconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]


    if model_type.lower() == 'hps' or model_type.lower() == 'dfcnn':
        if len(model_hyperpara['kernel_sizes']) == 8:
            if test_type%10 == 0:
                model_hyperpara['conv_sharing'] = [True, True, True, True]
            elif test_type%10 == 1:
                model_hyperpara['conv_sharing'] = [False, False, False, True]
            elif test_type%10 == 2:
                model_hyperpara['conv_sharing'] = [False, False, True, True]
            elif test_type%10 == 3:
                model_hyperpara['conv_sharing'] = [False, True, True, True]
            elif test_type%10 == 4:
                model_hyperpara['conv_sharing'] = [True, False, False, False]
            elif test_type%10 == 5:
                model_hyperpara['conv_sharing'] = [True, True, False, False]
            elif test_type%10 == 6:
                model_hyperpara['conv_sharing'] = [True, True, True, False]
            elif test_type%10 == 7:
                model_hyperpara['conv_sharing'] = [False, True, False, True]
            elif test_type%10 == 8:
                model_hyperpara['conv_sharing'] = [True, False, True, False]
            elif test_type%10 == 9:
                model_hyperpara['conv_sharing'] = [False, True, True, False]
        elif len(model_hyperpara['kernel_sizes']) == 12:
            if test_type%20 == 0:
                model_hyperpara['conv_sharing'] = [True, True, True, True, True, True]
            elif test_type%20 == 1:
                model_hyperpara['conv_sharing'] = [False, False, False, False, False, True]
            elif test_type%20 == 2:
                model_hyperpara['conv_sharing'] = [False, False, False, False, True, True]
            elif test_type%20 == 3:
                model_hyperpara['conv_sharing'] = [False, False, False, True, True, True]
            elif test_type%20 == 4:
                model_hyperpara['conv_sharing'] = [False, False, True, True, True, True]
            elif test_type%20 == 5:
                model_hyperpara['conv_sharing'] = [False, True, True, True, True, True]
            elif test_type%20 == 6:
                model_hyperpara['conv_sharing'] = [True, False, False, False, False, False]
            elif test_type%20 == 7:
                model_hyperpara['conv_sharing'] = [True, True, False, False, False, False]
            elif test_type%20 == 8:
                model_hyperpara['conv_sharing'] = [True, True, True, False, False, False]
            elif test_type%20 == 9:
                model_hyperpara['conv_sharing'] = [True, True, True, True, False, False]
            elif test_type%20 == 10:
                model_hyperpara['conv_sharing'] = [True, True, True, True, True, False]
            elif test_type%20 == 11:
                model_hyperpara['conv_sharing'] = [False, True, False, True, False, True]
            elif test_type%20 == 12:
                model_hyperpara['conv_sharing'] = [False, False, True, False, False, False]
            elif test_type%20 == 13:
                model_hyperpara['conv_sharing'] = [False, False, True, True, False, False]
            elif test_type%20 == 14:
                model_hyperpara['conv_sharing'] = [False, False, True, True, True, False]
            elif test_type%20 == 15:
                model_hyperpara['conv_sharing'] = [False, False, False, True, False, True]

    print(model_hyperpara['model_type'])
    if model_type.lower() == 'hps' or model_type.lower() == 'dfcnn':
        print("\tConv Sharing - ", model_hyperpara['conv_sharing'])

    return model_hyperpara

if __name__ == '__main__':
    tmp_game = getattr(l2agames, 'Pong')

    each_task_params, default_game_params = [], tmp_game._external_params.get_defaults()
    for env_cnt in range(5):
        new_params, temp = dict(default_game_params), tmp_game._external_params.sample()
        new_params['img/noise'] = temp['img/noise']
        new_params['img/rotation'] = temp['img/rotation']
        new_params['bg_color'] = temp['bg_color']
        each_task_params.append(new_params)

    write_params_to_csv(each_task_params, 'delete_this.csv')
    loaded_params = load_params_from_csv('delete_this.csv')
    print("End")
