import sys
import os
from math import sqrt
import pickle

import scipy.io as spio
import numpy as np
from tensorflow import reset_default_graph

from classification.train import train_mtl, train_lifelong, train_den_net, train_ewc
from classification.train_selective_transfer import train_lifelong_LASEM, train_lifelong_LASEM_fixedProb, train_lifelong_DARTS_hybridNN
from classification.train_bruteforce_configsearch import train_lifelong_bruteforce_config_search
from classification.train_semisupervised import train_semilifelong
from classification.gen_data import data_handler_for_experiment

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
RESULT_ROOT = os.path.join(PROJECT_ROOT, 'Result')

############################################
#### functions to load training results ####
############################################
def loadmat(filename):
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

#### checks if entries in dictionary are mat-objects
def _check_keys(dict):
    for key in dict:
        if isinstance(dict[key], np.ndarray):
            if isinstance(dict[key][0], spio.matlab.mio5_params.mat_struct):
                #### cell of structure case
                tmp = np.zeros((len(dict[key]),), dtype=np.object)
                for cnt in range(len(dict[key])):
                    if sys.version_info.major < 3:
                        tmp[cnt] = _todict2(dict[key][cnt])
                    else:
                        tmp[cnt] = _todict3(dict[key][cnt])
                dict[key] = tmp
            else:
                #### just array case
                dict[key] = list(dict[key])
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            if sys.version_info.major < 3:
                dict[key] = _todict2(dict[key])
            else:
                dict[key] = _todict3(dict[key])
    return dict

#### recursive function constructing nested dictionaries from matobjects
def _todict2(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict2(elem)
        elif isinstance(elem, unicode):
            dict[strg] = str(elem)
        elif isinstance(elem, np.ndarray):
            cell_struct = False
            if len(elem) > 0:
                if isinstance(elem[0], spio.matlab.mio5_params.mat_struct):
                    cell_struct = True
            if cell_struct:
                #### cell of structure case
                tmp = np.zeros((len(elem),), dtype=np.object)
                for cnt in range(len(elem)):
                    tmp[cnt] = _todict2(elem[cnt])
                dict[strg] = tmp
            else:
                #### just array case
                dict[strg] = list(elem)
        elif strg == 'hidden_layer':
            dict[strg] = [elem]
        else:
            dict[strg] = elem
    return dict

def _todict3(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict3(elem)
        elif isinstance(elem, str):
            dict[strg] = str(elem)
        elif isinstance(elem, np.ndarray):
            cell_struct = False
            if len(elem) > 0:
                if isinstance(elem[0], spio.matlab.mio5_params.mat_struct):
                    cell_struct = True
            if cell_struct:
                #### cell of structure case
                tmp = np.zeros((len(elem),), dtype=np.object)
                for cnt in range(len(elem)):
                    tmp[cnt] = _todict3(elem[cnt])
                dict[strg] = tmp
            else:
                #### just array case
                dict[strg] = list(elem)
        elif strg == 'hidden_layer':
            dict[strg] = [elem]
        else:
            dict[strg] = elem
    return dict

#### functions to generate directories to save weak labeler model files -- only for DP-SSCL
def gen_dir_to_save_weak_labelers(path_to_wls_root, result_name, exp_type):
    if 'cifar10_mako' in exp_type:
        assert (result_name not in os.listdir(path_to_wls_root)), "Weak labelers exist!"
        new_result_path = os.path.join(path_to_wls_root, result_name)
        os.mkdir(new_result_path)
        for task_cnt1 in range(9):
            for task_cnt2 in range(task_cnt1+1, 10):
                new_dir_name = '%s_%s'%(task_cnt1, task_cnt2)
                os.mkdir(os.path.join(new_result_path, new_dir_name))
    else:
        raise NotImplementedError("Not yet implemented!")

#### compare two model_info_summary
#### cannot check whether two models are equal because floating number in Python and MATLABare different
def check_model_equivalency(model_info1, model_info2):
    check_keys1 = [(model_info1_elem in model_info2) for model_info1_elem in model_info1]
    check_keys2 = [(model_info2_elem in model_info1) for model_info2_elem in model_info2]
    if all(check_keys1) and all(check_keys2):
        return all([(model_info1[key] == model_info2[key]) for key in model_info1])
    else:
        return False

def print_model_info(model_dict):
    key_list = list(model_dict.keys())
    key_list.sort()
    for key in key_list:
        print(key + " : ", model_dict[key])

############################################
#### functions to save training results ####
############################################
def mean_of_list(list_input):
    return float(sum(list_input))/len(list_input)

def stddev_of_list(list_input):
    list_mean = mean_of_list(list_input)
    sq_err = [(x-list_mean)**2 for x in list_input]

    if len(list_input)<2:
        return 0.0
    else:
        return sqrt(sum(sq_err)/float(len(list_input)-1))


def model_info_summary(model_architecture, model_hyperpara, train_hyperpara, num_para_in_model=-1):
    tmp_dict = {}
    tmp_dict['architecture'] = model_architecture
    tmp_dict['learning_rate'] = train_hyperpara['lr']
    tmp_dict['improvement_threshold'] = train_hyperpara['improvement_threshold']
    tmp_dict['early_stopping_para'] = [train_hyperpara['patience'], train_hyperpara['patience_multiplier']]

    for model_hyperpara_elem in model_hyperpara:
        if 'ffnn' in model_architecture.lower() and not ((model_hyperpara_elem == 'kernel_sizes') or (model_hyperpara_elem == 'channel_sizes') or (model_hyperpara_elem == 'stride_sizes') or (model_hyperpara_elem == 'max_pooling') or (model_hyperpara_elem == 'pooling_size') or (model_hyperpara_elem == 'padding_type') or (model_hyperpara_elem == 'dropout')):
            tmp_dict[model_hyperpara_elem] = model_hyperpara[model_hyperpara_elem]
        elif 'cnn' in model_architecture.lower():
            tmp_dict[model_hyperpara_elem] = model_hyperpara[model_hyperpara_elem]

    if num_para_in_model > -1:
        tmp_dict['number_of_trainable_parameters'] = num_para_in_model

    if train_hyperpara['stl_analysis']:
        tmp_dict['STL_analysis'] = True
        tmp_dict['STL_analysis_task'] = train_hyperpara['stl_task_to_learn']
        tmp_dict['STL_analysis_ratio'] = train_hyperpara['stl_total_data_ratio']
        tmp_dict['STL_analysis_comparable_epoch'] = 1+train_hyperpara['patience']*train_hyperpara['stl_task_to_learn']

    # parameters related to semi-supervised learning
    if 'task_similarity_score_threshold' in train_hyperpara:
        tmp_dict['task_similarity_score_threshold'] = train_hyperpara['task_similarity_score_threshold']
    if 'sample_lf_from_earlier_task_prob' in train_hyperpara:
        tmp_dict['prob_sampling_earlier_lfs'] = train_hyperpara['sample_lf_from_earlier_task_prob']
    if 'strong_labeler' in train_hyperpara:
        tmp_dict['labelers_ensemble_method'] = train_hyperpara['strong_labeler']
    if 'num_unlabeled_data' in train_hyperpara:
        tmp_dict['number_unlabeled_training_data'] = train_hyperpara['num_unlabeled_data']

    if 'num_weak_labelers_min' in train_hyperpara:
        tmp_dict['number_weak_labelers_min'] = train_hyperpara['num_weak_labelers_min']
    if 'num_weak_labelers_max' in train_hyperpara:
        tmp_dict['number_weak_labelers_max'] = train_hyperpara['num_weak_labelers_max']
    if 'num_weak_labelers_keep' in train_hyperpara:
        tmp_dict['number_weak_labelers_keep'] = train_hyperpara['num_weak_labelers_keep']
    return tmp_dict


def reformat_result_for_mat(model_architecture, model_hyperpara, train_hyperpara, result_from_train_run, data_group_list, num_para_in_model=-1, doLifelong=False, task_order=None):
    result_of_curr_run = {}
    #### 'model_specific_info' element
    result_of_curr_run['model_specific_info'] = model_info_summary(model_architecture, model_hyperpara, train_hyperpara, num_para_in_model)
    #result_of_curr_run['task_order'] = np.array(task_order)

    num_run_per_model, best_valid_error_list, best_test_error_list, train_time_list = len(result_from_train_run), [], [], []
    result_of_curr_run['result_of_each_run'] = np.zeros((num_run_per_model,), dtype=np.object)
    if 'transferred_labelers_info' in result_from_train_run[0]:
        result_of_curr_run['transferred_labelers_info'] = np.zeros((num_run_per_model,), dtype=np.object)
    if 'labelers_converged_epochs' in result_from_train_run[0]:
        result_of_curr_run['labelers_converged_epochs'] = np.zeros((num_run_per_model,), dtype=np.object)
    if 'snuba_iterations' in result_from_train_run[0]:
        result_of_curr_run['snuba_iterations'] = np.zeros((num_run_per_model,), dtype=np.object)
    if 'makoint_LEEPscore' in result_from_train_run[0]:
        result_of_curr_run['makoint_LEEPscore'] = np.zeros((num_run_per_model,), dtype=np.object)
    if doLifelong:
        best_test_error_each_task_list, test_error_at_last_epoch_list = [], []
        best_test_error_each_task_avg_list, test_error_at_last_epoch_avg_list = [], []

    for cnt in range(num_run_per_model):
        result_of_curr_run['result_of_each_run'][cnt] = result_from_train_run[cnt]
        train_time_list.append(result_from_train_run[cnt]['training_time'])
        if doLifelong:
            #chk_epoch = [x-1 for x in result_from_train_run[cnt]['task_changed_epoch'][1:]]+[result_from_train_run[cnt]['num_epoch']]
            chk_epoch = [x-1 for x in result_from_train_run[cnt]['task_changed_epoch'][1:]]
            best_test_error_each_task_list.append([result_from_train_run[cnt]['history_best_test_error'][x] for x in chk_epoch] + [result_from_train_run[cnt]['history_best_test_error'][-1]])
            best_test_error_each_task_avg_list.append(mean_of_list(best_test_error_each_task_list[cnt]))

            test_error_at_last_epoch_list.append(result_from_train_run[cnt]['history_test_error'][-1][0:-1])
            test_error_at_last_epoch_avg_list.append(mean_of_list(test_error_at_last_epoch_list[cnt]))
        else:
            best_valid_error_list.append(result_from_train_run[cnt]['best_validation_error'])
            best_test_error_list.append(result_from_train_run[cnt]['test_error_at_best_epoch'])

        if 'transferred_labelers_info' in result_from_train_run[0]:
            result_of_curr_run['transferred_labelers_info'][cnt] = result_from_train_run[cnt]['transferred_labelers_info']
        if 'labelers_converged_epochs' in result_from_train_run[0]:
            result_of_curr_run['labelers_converged_epochs'][cnt] = result_from_train_run[cnt]['labelers_converged_epochs']
        if 'snuba_iterations' in result_from_train_run[0]:
            result_of_curr_run['snuba_iterations'][cnt] = result_from_train_run[cnt]['snuba_iterations']
        if 'makoint_LEEPscore' in result_from_train_run[0]:
            result_of_curr_run['makoint_LEEPscore'][cnt] = result_from_train_run[cnt]['makoint_LEEPscore']

    result_of_curr_run['training_time'] = train_time_list
    result_of_curr_run['training_time_mean'] = mean_of_list(train_time_list)
    result_of_curr_run['training_time_stddev'] = stddev_of_list(train_time_list)
    result_of_curr_run['train_valid_data_group'] = data_group_list
    if doLifelong:
        result_of_curr_run['best_test_error_each_task'] = best_test_error_each_task_list
        result_of_curr_run['best_test_error_each_task_mean'] = mean_of_list(best_test_error_each_task_avg_list)
        result_of_curr_run['best_test_error_each_task_std'] = stddev_of_list(best_test_error_each_task_avg_list)
        result_of_curr_run['test_error_at_last_epoch'] = test_error_at_last_epoch_list
        result_of_curr_run['test_error_at_last_epoch_mean'] = mean_of_list(test_error_at_last_epoch_avg_list)
        result_of_curr_run['test_error_at_last_epoch_std'] = stddev_of_list(test_error_at_last_epoch_avg_list)
        if task_order is not None:
            result_of_curr_run['result_of_each_run'][-1]['task_order'] = np.array(task_order)
    else:
        result_of_curr_run['best_valid_error'] = best_valid_error_list
        result_of_curr_run['best_valid_error_mean'] = mean_of_list(best_valid_error_list)
        result_of_curr_run['best_valid_error_stddev'] = stddev_of_list(best_valid_error_list)
        result_of_curr_run['best_test_error'] = best_test_error_list
        result_of_curr_run['best_test_error_mean'] = mean_of_list(best_test_error_list)
        result_of_curr_run['best_test_error_stddev'] = stddev_of_list(best_test_error_list)

    return result_of_curr_run


############################################
#### functions to run several training  ####
####       with same model setting      ####
############################################
def train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, mat_file_name, classification_prob, saved_result=None, useGPU=False, GPU_device=0, doLifelong=False, doSemiLifelong=False, saveParam=False, saveParamDir=None, saveGraph=False, tfInitParamPath=None, task_order=None, debug_data_group=-1, saveResultPkl=False):
    max_run_cnt = train_hyperpara['num_run_per_model']

    #### process results of previous training
    if (saved_result is None) and not (os.path.isfile('./Result/'+mat_file_name)):
        saved_result = np.zeros((1,), dtype=np.object)
    elif (saved_result is None):
        saved_result_tmp = loadmat('./Result/'+mat_file_name)
        ##### for python 3.7.5/scipy 1.3.1(py37h7c811a0), need to use saved_result_tmp['training_summary'][0]
        num_prev_test_model = len(saved_result_tmp['training_summary'])
        saved_result = np.zeros((num_prev_test_model+1,), dtype=np.object)
        for cnt in range(num_prev_test_model):
            saved_result[cnt] = saved_result_tmp['training_summary'][cnt]
    else:
        num_prev_result, prev_result_tmp = len(saved_result), saved_result
        saved_result = np.zeros((num_prev_result+1,), dtype=np.object)
        for cnt in range(num_prev_result):
            saved_result[cnt] = prev_result_tmp[cnt]

    #### process results saved in temporary file
    if not (os.path.isfile('./Result/temp_'+mat_file_name)):
        temp_result_exist = False
    else:
        saved_temp_result_tmp = loadmat('./Result/temp_'+mat_file_name)
        saved_temp_result = saved_temp_result_tmp['training_summary_temp']
        print('\nModel in temp file')
        print_model_info(saved_temp_result['model_specific_info'])
        print('\nModel for current train')
        print_model_info(model_info_summary(model_architecture, model_hyperpara, train_hyperpara))
        #use_temp_result = input('\nUse temp result? (T/F): ')
        use_temp_result = 't'
        if (use_temp_result is 'T') or (use_temp_result is 't') or (use_temp_result is 'True') or (use_temp_result is 'true'):
            temp_result_exist = True
        else:
            temp_result_exist = False

    result_from_train_run = []
    seed_group = train_hyperpara['train_valid_data_group']
    if temp_result_exist:
        if not (isinstance(saved_temp_result['train_valid_data_group'], np.ndarray) or isinstance(saved_temp_result['train_valid_data_group'], list)):
            result_from_train_run.append(saved_temp_result['result_of_each_run'])
        else:
            for elem in saved_temp_result['result_of_each_run']:
                result_from_train_run.append(elem)
        run_cnt_init = len(result_from_train_run)
        print("\nTemporary Result Exists! Start from %d\n" %(run_cnt_init+1))
    else:
        run_cnt_init = 0
        print("\nTemporary Result is Discarded! Overwrite Temp File\n")

    is_multiple_task_orders = False
    if isinstance(task_order, list):
        if isinstance(task_order[0], list):
            is_multiple_task_orders = True

    #### run training procedure with different dataset
    for run_cnt in range(run_cnt_init, max_run_cnt):
        # start debug code for DEN experiments terminating early
        if max_run_cnt == 1 and debug_data_group > -1:
            run_cnt = debug_data_group

        if ('omniglot' in data_type) or ('officehome' in data_type) or ('cifar100_l2m' in data_type):
            train_data, valid_data, test_data = dataset[0][seed_group[run_cnt]], dataset[1][seed_group[run_cnt]], dataset[2][seed_group[run_cnt]]
        else:
            train_data, valid_data, test_data = dataset[0][seed_group[run_cnt]], dataset[1][seed_group[run_cnt]], dataset[2]

        save_param_path = None
        if saveParam:
            save_param_path = saveParamDir+'/run'+str(run_cnt)
            os.mkdir(save_param_path)

        if ('tf' in model_architecture) and (tfInitParamPath is not None):
            assert ('best_model_parameter.mat' in os.listdir(tfInitParamPath+'/run'+str(run_cnt))), "(TensorFactor) Parameter file to load does not exist!"
            param_file_to_load = tfInitParamPath+'/run'+str(run_cnt)+'/best_model_parameter.mat'
            tf_init_params = loadmat(param_file_to_load)['parameter']
            print("\tLoad parameter file to initialize TensorFactor model\n")
        else:
            tf_init_params = None

        if is_multiple_task_orders:
            task_order_this_run = task_order[run_cnt]
        elif (task_order is None) and ('task_order' in train_hyperpara.keys()):
            task_order_this_run = train_hyperpara['task_order']
        else:
            task_order_this_run = task_order

        print("Training/Validation data group : %d\n" %(seed_group[run_cnt]))
        if ('den' in model_architecture) or ('dynamically' in model_architecture):
            train_result_tmp, num_model_para = train_den_net(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, classification_prob, doLifelong, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1), task_order=task_order_this_run)
        elif ('ewc' in model_architecture) or ('elastic' in model_architecture):
            train_result_tmp, num_model_para = train_ewc(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, classification_prob, doLifelong, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1))
        elif ('lasem' in model_architecture) and ('fixed' not in model_architecture):
            train_result_tmp, num_model_para = train_lifelong_LASEM(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, classification_prob, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1), tfInitParam=tf_init_params, run_cnt=run_cnt)
        elif (model_architecture == 'lasem_fixed_dfcnn'):
            train_result_tmp, num_model_para = train_lifelong_LASEM_fixedProb(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, classification_prob, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1), tfInitParam=tf_init_params, run_cnt=run_cnt)
        elif ('darts' in model_architecture):
            train_result_tmp, num_model_para = train_lifelong_DARTS_hybridNN(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, classification_prob, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1), tfInitParam=tf_init_params, run_cnt=run_cnt)
        elif ('bruteforce' in model_architecture):
            train_result_tmp, num_model_para = train_lifelong_bruteforce_config_search(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1), run_cnt=run_cnt)
        elif doSemiLifelong:
            if not 'DPSSCL_wls' in os.listdir(RESULT_ROOT):
                os.mkdir(os.path.join(RESULT_ROOT, 'DPSSCL_wls'))
            mat_file_name_wo_ext = mat_file_name[:mat_file_name.rfind('.')]
            gen_dir_to_save_weak_labelers(os.path.join(RESULT_ROOT, 'DPSSCL_wls'), "%s_r%d"%(mat_file_name_wo_ext, run_cnt), data_type)
            WLS_ROOT = os.path.join(RESULT_ROOT, 'DPSSCL_wls', "%s_r%d"%(mat_file_name_wo_ext, run_cnt))
            train_result_tmp, num_model_para = train_semilifelong(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, classification_prob, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1), run_cnt=run_cnt, task_order=task_order_this_run, weak_labeler_root=WLS_ROOT)
        elif doLifelong:
            train_result_tmp, num_model_para = train_lifelong(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, classification_prob, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1), tfInitParam=tf_init_params, run_cnt=run_cnt, task_order=task_order_this_run)
        else:
            train_result_tmp, num_model_para = train_mtl(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, classification_prob, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1), tfInitParam=tf_init_params, run_cnt=run_cnt)
        result_from_train_run.append(train_result_tmp)

        if run_cnt < max_run_cnt-1:
            result_of_curr_run = reformat_result_for_mat(model_architecture, model_hyperpara, train_hyperpara, result_from_train_run, seed_group[0:run_cnt+1], num_model_para, doLifelong or doSemiLifelong, task_order=task_order_this_run)
            spio.savemat('./Result/temp_'+mat_file_name, {'training_summary_temp':result_of_curr_run})
        elif os.path.isfile('./Result/temp_'+mat_file_name):
            os.remove('./Result/temp_'+mat_file_name)

        print("%d-th training run complete\n\n" % (run_cnt+1))
        reset_default_graph()

    #### save training summary
    result_of_curr_run = reformat_result_for_mat(model_architecture, model_hyperpara, train_hyperpara, result_from_train_run, seed_group[0:max_run_cnt], num_model_para, doLifelong or doSemiLifelong, task_order=task_order_this_run)
    saved_result[-1] = result_of_curr_run
    spio.savemat('./Result/'+mat_file_name, {'training_summary':saved_result})
    # Save pickle results (no postdoc MATLAB license from SEAS at UPenn)
    if saveResultPkl:
        with open('Result/' + mat_file_name[:-4] + '.pkl', 'wb') as f:
            pickle.dump(result_from_train_run, f)

    return saved_result


def train_run_for_each_model_v2(model_architecture, model_hyperpara, train_hyperpara, exp_design, data_type, mat_file_name, classification_prob, saved_result=None, useGPU=False, GPU_device=0, doLifelong=False, saveParam=False, saveParamDir=None, saveGraph=False, tfInitParamPath=None, task_order=None):
    max_run_cnt = train_hyperpara['num_run_per_model']
    categorized_train_data, categorized_test_data, experiments_design = exp_design

    #### process results of previous training
    if (saved_result is None) and not (os.path.isfile('./Result/'+mat_file_name)):
        saved_result = np.zeros((1,), dtype=np.object)
    elif (saved_result is None):
        saved_result_tmp = loadmat('./Result/'+mat_file_name)
        ##### for python 3.7.5/scipy 1.3.1(py37h7c811a0), need to use saved_result_tmp['training_summary'][0]
        num_prev_test_model = len(saved_result_tmp['training_summary'])
        saved_result = np.zeros((num_prev_test_model+1,), dtype=np.object)
        for cnt in range(num_prev_test_model):
            saved_result[cnt] = saved_result_tmp['training_summary'][cnt]
    else:
        num_prev_result, prev_result_tmp = len(saved_result), saved_result
        saved_result = np.zeros((num_prev_result+1,), dtype=np.object)
        for cnt in range(num_prev_result):
            saved_result[cnt] = prev_result_tmp[cnt]

    #### process results saved in temporary file
    if not (os.path.isfile('./Result/temp_'+mat_file_name)):
        temp_result_exist = False
    else:
        saved_temp_result_tmp = loadmat('./Result/temp_'+mat_file_name)
        saved_temp_result = saved_temp_result_tmp['training_summary_temp']
        print('\nModel in temp file')
        print_model_info(saved_temp_result['model_specific_info'])
        print('\nModel for current train')
        print_model_info(model_info_summary(model_architecture, model_hyperpara, train_hyperpara))
        #use_temp_result = input('\nUse temp result? (T/F): ')
        use_temp_result = 't'
        if (use_temp_result is 'T') or (use_temp_result is 't') or (use_temp_result is 'True') or (use_temp_result is 'true'):
            temp_result_exist = True
        else:
            temp_result_exist = False

    result_from_train_run = []
    seed_group = train_hyperpara['train_valid_data_group']
    if temp_result_exist:
        if not (isinstance(saved_temp_result['train_valid_data_group'], np.ndarray) or isinstance(saved_temp_result['train_valid_data_group'], list)):
            result_from_train_run.append(saved_temp_result['result_of_each_run'])
        else:
            for elem in saved_temp_result['result_of_each_run']:
                result_from_train_run.append(elem)
        run_cnt_init = len(result_from_train_run)
        print("\nTemporary Result Exists! Start from %d\n" %(run_cnt_init+1))
    else:
        run_cnt_init = 0
        print("\nTemporary Result is Discarded! Overwrite Temp File\n")

    is_multiple_task_orders = False
    if isinstance(task_order, list):
        if isinstance(task_order[0], list):
            is_multiple_task_orders = True

    #### run training procedure with different dataset
    for run_cnt in range(run_cnt_init, max_run_cnt):
        train_data, valid_data, test_data = data_handler_for_experiment(categorized_train_data, categorized_test_data, experiments_design[seed_group[run_cnt]], model_hyperpara['image_dimension'])

        save_param_path = None
        if saveParam:
            save_param_path = saveParamDir+'/run'+str(run_cnt)
            os.mkdir(save_param_path)

        if ('tf' in model_architecture) and (tfInitParamPath is not None):
            assert ('best_model_parameter.mat' in os.listdir(tfInitParamPath+'/run'+str(run_cnt))), "(TensorFactor) Parameter file to load does not exist!"
            param_file_to_load = tfInitParamPath+'/run'+str(run_cnt)+'/best_model_parameter.mat'
            tf_init_params = loadmat(param_file_to_load)['parameter']
            print("\tLoad parameter file to initialize TensorFactor model\n")
        else:
            tf_init_params = None

        if is_multiple_task_orders:
            task_order_this_run = task_order[run_cnt]
        elif (task_order is None) and ('task_order' in train_hyperpara.keys()):
            task_order_this_run = train_hyperpara['task_order']
        else:
            task_order_this_run = task_order

        print("Training/Validation data group : %d\n" %(seed_group[run_cnt]))
        if ('den' in model_architecture) or ('dynamically' in model_architecture):
            train_result_tmp, num_model_para = train_den_net(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, classification_prob, doLifelong, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1), task_order=task_order_this_run)
        elif ('ewc' in model_architecture) or ('elastic' in model_architecture):
            train_result_tmp, num_model_para = train_ewc(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, classification_prob, doLifelong, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1))
        elif ('lasem' in model_architecture) and ('fixed' not in model_architecture):
            train_result_tmp, num_model_para = train_lifelong_LASEM(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, classification_prob, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1), tfInitParam=tf_init_params, run_cnt=run_cnt)
        elif (model_architecture == 'lasem_fixed_dfcnn'):
            train_result_tmp, num_model_para = train_lifelong_LASEM_fixedProb(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, classification_prob, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1), tfInitParam=tf_init_params, run_cnt=run_cnt)
        elif ('bruteforce' in model_architecture):
            train_result_tmp, num_model_para = train_lifelong_bruteforce_config_search(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1), run_cnt=run_cnt)
        elif doLifelong:
            train_result_tmp, num_model_para = train_lifelong(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, classification_prob, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1), tfInitParam=tf_init_params, run_cnt=run_cnt, task_order=task_order_this_run)
        else:
            train_result_tmp, num_model_para = train_mtl(model_architecture, model_hyperpara, train_hyperpara, [train_data, valid_data, test_data], data_type, classification_prob, useGPU, GPU_device, save_param=saveParam, param_folder_path=save_param_path, save_graph=(saveGraph and run_cnt < 1), tfInitParam=tf_init_params, run_cnt=run_cnt)
        result_from_train_run.append(train_result_tmp)

        if run_cnt < max_run_cnt-1:
            result_of_curr_run = reformat_result_for_mat(model_architecture, model_hyperpara, train_hyperpara, result_from_train_run, seed_group[0:run_cnt+1], num_model_para, doLifelong, task_order=task_order_this_run)
            spio.savemat('./Result/temp_'+mat_file_name, {'training_summary_temp':result_of_curr_run})
        elif os.path.isfile('./Result/temp_'+mat_file_name):
            os.remove('./Result/temp_'+mat_file_name)

        print("%d-th training run complete\n\n" % (run_cnt+1))
        reset_default_graph()

    #### save training summary
    result_of_curr_run = reformat_result_for_mat(model_architecture, model_hyperpara, train_hyperpara, result_from_train_run, seed_group[0:max_run_cnt], num_model_para, doLifelong, task_order=task_order_this_run)
    saved_result[-1] = result_of_curr_run
    spio.savemat('./Result/'+mat_file_name, {'training_summary':saved_result})

    return saved_result
