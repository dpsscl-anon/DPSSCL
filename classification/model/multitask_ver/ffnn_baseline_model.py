import numpy as np
import tensorflow as tf

from utils.utils import *
from utils.utils_nn import *

#################################################
############ Simple Feedforward Net #############
#################################################
#### Feedforward Neural Net
class FFNN_batch():
    def __init__(self, dim_layers, input_size, learning_rate, learning_rate_decay, classification=False):
        self.num_layers = len(dim_layers)
        self.layers_size = dim_layers
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        if type(input_size) == list:
            tmp_size = 1
            for elem in input_size:
                tmp_size *= elem
            self.input_size = tmp_size
        else:
            self.input_size = input_size

        #### placeholder of model
        self.model_input = new_placeholder([None, self.input_size])
        self.true_output = new_placeholder([None])
        self.epoch = tf.placeholder(dtype=tf.float32)

        #### layers of model
        if classification:
            self.layers, self.param = new_fc_net(self.model_input, self.layers_size, params=None, output_type='classification')
        else:
            self.layers, self.param = new_fc_net(self.model_input, self.layers_size, params=None, output_type=None)

        #### functions of model
        if classification and self.layers_size[-1]>1:
            self.eval = tf.nn.softmax(self.layers[-1])
            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.true_output, logits=self.layers[-1])
            self.accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.eval, 1), tf.argmax(self.true_output, 1)), tf.float32))
        else:
            self.eval = self.layers[-1]
            self.loss = 2.0* tf.nn.l2_loss(self.eval-self.true_output)
            if classification:
                self.accuracy = tf.reduce_sum(tf.cast(tf.equal((self.eval>0.5), (self.true_output>0.5)), tf.float32))

        #self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.loss)
        self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.loss)

        self.num_trainable_var = count_trainable_var()


#### Feedforward Neural Network - mini batch ver.
class FFNN_minibatch():
    def __init__(self, dim_layers, input_size, batch_size, learning_rate, learning_rate_decay, data_list, classification=False, saveData=False):
        self.num_layers = len(dim_layers)
        self.layers_size = dim_layers
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size
        if type(input_size) == list:
            tmp_size = 1
            for elem in input_size:
                tmp_size *= elem
            self.input_size = tmp_size
        else:
            self.input_size = input_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)

        if saveData:
            self.data = tflized_data(data_list, do_MTL=False)
            self.data_index = tf.placeholder(dtype=tf.int32)
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data(self.data, self.batch_size, self.data_index, do_MTL=False)
        else:
            self.model_input = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size])
            self.true_output = tf.placeholder(dtype=tf.float32, shape=[self.batch_size])
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data_not_in_gpu(self.model_input, self.true_output, do_MTL=False)

        #### layers of model for train data
        if classification:
            self.train_layers, self.param = new_fc_net(train_x_batch, self.layers_size, params=None, output_type='classification')
        else:
            self.train_layers, self.param = new_fc_net(train_x_batch, self.layers_size, params=None, output_type=None)

        #### layers of model for validation data
        if classification:
            self.valid_layers, _ = new_fc_net(valid_x_batch, self.layers_size, params=self.param, output_type='classification')
        else:
            self.valid_layers, _ = new_fc_net(valid_x_batch, self.layers_size, params=self.param, output_type=None)

        #### layers of model for test data
        if classification:
            self.test_layers, _ = new_fc_net(test_x_batch, self.layers_size, params=self.param, output_type='classification')
        else:
            self.test_layers, _ = new_fc_net(test_x_batch, self.layers_size, params=self.param, output_type=None)

        #### functions of model
        tr_eval, v_eval, test_eval, tr_loss, v_loss, test_loss, tr_acc, v_acc, test_acc = mtl_model_output_functions(models=[[self.train_layers], [self.valid_layers], [self.test_layers]], y_batches=[[train_y_batch], [valid_y_batch], [test_y_batch]], num_tasks=1, dim_output=self.layers_size[-1], classification=classification)

        self.train_eval, self.valid_eval, self.test_eval = tr_eval[0], v_eval[0], test_eval[0]
        self.train_loss, self.valid_loss, self.test_loss = tr_loss[0], v_loss[0], test_loss[0]
        if not (tr_acc is None):
            self.train_accuracy, self.valid_accuracy, self.test_accuracy = tr_acc[0], v_acc[0], test_acc[0]

        #self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.train_loss)
        self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss)

        self.num_trainable_var = count_trainable_var()


########################################################
####   Single task learner for Multi-task Learning  ####
########################################################
#### FFNN3 model for MTL
class MTL_several_FFNN_minibatch():
    def __init__(self, num_tasks, dim_layers, input_size, batch_size, learning_rate, learning_rate_decay, data_list, classification=False, saveData=False):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_layers)
        self.layers_size = dim_layers
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size
        if type(input_size) == list:
            tmp_size = 1
            for elem in input_size:
                tmp_size *= elem
            self.input_size = tmp_size
        else:
            self.input_size = input_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)

        if saveData:
            self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)
            self.data_index = tf.placeholder(dtype=tf.int32)
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data(self.data, self.batch_size, self.data_index, do_MTL=True)
        else:
            self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size]) for _ in range(self.num_tasks)]
            self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data_not_in_gpu(self.model_input, self.true_output, do_MTL=True)

        #### layers of model for train data
        self.train_models = []
        self.param, self.fc_param = [], []
        for task_cnt in range(self.num_tasks):
            if classification:
                model_tmp, param_tmp = new_fc_net(train_x_batch[task_cnt], self.layers_size, params=None, output_type='classification')
            else:
                model_tmp, param_tmp = new_fc_net(train_x_batch[task_cnt], self.layers_size, params=None, output_type=None)
            self.train_models.append(model_tmp)
            self.fc_param.append(param_tmp)
            self.param = self.param + param_tmp

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            if classification:
                model_tmp, _ = new_fc_net(valid_x_batch[task_cnt], self.layers_size, params=self.fc_param[task_cnt], output_type='classification')
            else:
                model_tmp, _ = new_fc_net(valid_x_batch[task_cnt], self.layers_size, params=self.fc_param[task_cnt], output_type=None)
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            if classification:
                model_tmp, _ = new_fc_net(test_x_batch[task_cnt], self.layers_size, params=self.fc_param[task_cnt], output_type='classification')
            else:
                model_tmp, _ = new_fc_net(test_x_batch[task_cnt], self.layers_size, params=self.fc_param[task_cnt], output_type=None)
            self.test_models.append(model_tmp)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], num_tasks, self.layers_size[-1], classification)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()


########################################################
#### Single Feedforward Net for Multi-task Learning ####
########################################################
#### FFNN3 model for MTL
class MTL_FFNN_minibatch():
    def __init__(self, num_tasks, dim_layers, input_size, batch_size, learning_rate, learning_rate_decay, data_list, classification=False, saveData=False):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_layers)
        self.layers_size = dim_layers
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size
        if type(input_size) == list:
            tmp_size = 1
            for elem in input_size:
                tmp_size *= elem
            self.input_size = tmp_size
        else:
            self.input_size = input_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)

        if saveData:
            self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)
            self.data_index = tf.placeholder(dtype=tf.int32)
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data(self.data, self.batch_size, self.data_index, do_MTL=True)
        else:
            self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size]) for _ in range(self.num_tasks)]
            self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data_not_in_gpu(self.model_input, self.true_output, do_MTL=True)

        #### layers of model for train data
        self.train_models = []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0 and classification:
                model_tmp, self.param = new_fc_net(train_x_batch[task_cnt], self.layers_size, params=None, output_type='classification')
            elif task_cnt == 0:
                model_tmp, self.param = new_fc_net(train_x_batch[task_cnt], self.layers_size, params=None, output_type=None)
            elif classification:
                model_tmp, _ = new_fc_net(train_x_batch[task_cnt], self.layers_size, params=self.param, output_type='classification')
            else:
                model_tmp, _ = new_fc_net(train_x_batch[task_cnt], self.layers_size, params=self.param, output_type=None)
            self.train_models.append(model_tmp)

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            if classification:
                model_tmp, _ = new_fc_net(valid_x_batch[task_cnt], self.layers_size, params=self.param, output_type='classification')
            else:
                model_tmp, _ = new_fc_net(valid_x_batch[task_cnt], self.layers_size, params=self.param, output_type=None)
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            if classification:
                model_tmp, _ = new_fc_net(test_x_batch[task_cnt], self.layers_size, params=self.param, output_type='classification')
            else:
                model_tmp, _ = new_fc_net(test_x_batch[task_cnt], self.layers_size, params=self.param, output_type=None)
            self.test_models.append(model_tmp)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], num_tasks, self.layers_size[-1], classification)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()


########################################################
#### Single Feedforward Net for Multi-task Learning ####
########         Hard Parameter Sharing         ########
########################################################
class MTL_FFNN_HPS_minibatch():
    def __init__(self, num_tasks, dim_layers, input_size, batch_size, learning_rate, learning_rate_decay, data_list, classification=False, saveData=False):
        self.num_tasks = num_tasks
        self.shared_layers_size = dim_layers[0]
        self.task_specific_layers_size = dim_layers[1]
        self.num_layers = [len(self.shared_layers_size)] + [len(self.task_specific_layers_size[x]) for x in range(self.num_tasks)]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size
        if type(input_size) == list:
            tmp_size = 1
            for elem in input_size:
                tmp_size *= elem
            self.input_size = tmp_size
        else:
            self.input_size = input_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)

        if saveData:
            self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)
            self.data_index = tf.placeholder(dtype=tf.int32)
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data(self.data, self.batch_size, self.data_index, do_MTL=True)
        else:
            self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size]) for _ in range(self.num_tasks)]
            self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data_not_in_gpu(self.model_input, self.true_output, do_MTL=True)

        #### layers of model for train data
        if classification:
            self.train_models, self.shared_param, self.specific_param = new_hardparam_fc_fc_nets(train_x_batch, self.shared_layers_size, self.task_specific_layers_size, self.num_tasks, shared_params=None, specific_params=None, output_type='classification')
        else:
            self.train_models, self.shared_param, self.specific_param = new_hardparam_fc_fc_nets(train_x_batch, self.shared_layers_size, self.task_specific_layers_size, self.num_tasks, shared_params=None, specific_params=None, output_type=None)
        self.param = self.shared_param + self.specific_param

        #### layers of model for validation data
        if classification:
            self.valid_models, _, _ = new_hardparam_fc_fc_nets(valid_x_batch, self.shared_layers_size, self.task_specific_layers_size, self.num_tasks, shared_params=self.shared_param, specific_params=self.specific_param, output_type='classification')
        else:
            self.valid_models, _, _ = new_hardparam_fc_fc_nets(valid_x_batch, self.shared_layers_size, self.task_specific_layers_size, self.num_tasks, shared_params=self.shared_param, specific_params=self.specific_param, output_type=None)

        #### layers of model for test data
        if classification:
            self.test_models, _, _ = new_hardparam_fc_fc_nets(test_x_batch, self.shared_layers_size, self.task_specific_layers_size, self.num_tasks, shared_params=self.shared_param, specific_params=self.specific_param, output_type='classification')
        else:
            self.test_models, _, _ = new_hardparam_fc_fc_nets(test_x_batch, self.shared_layers_size, self.task_specific_layers_size, self.num_tasks, shared_params=self.shared_param, specific_params=self.specific_param, output_type=None)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], num_tasks, self.task_specific_layers_size[0][-1], classification)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()


########################################################
######   Feedforward Net for Multi-task Learning   #####
######            Tensor Factorization             #####
########################################################
## data_list = [train_x, train_y, valid_x, valid_y, test_x, test_y]
##           or [ [(trainx_1, trainy_1), ..., (trainx_t, trainy_t)], [(validx_1, validy_1), ...], [(testx_1, testy_1), ...] ]
class MTL_FFNN_tensorfactor_minibatch():
    def __init__(self, num_tasks, dim_layers, input_size, batch_size, learning_rate, learning_rate_decay, data_list, classification=False, factor_type='Tucker', factor_eps_or_k=0.01, saveData=False):
        self.num_tasks = num_tasks
        self.shared_layers_size = dim_layers[0]
        self.task_specific_layers_size = dim_layers[1]
        self.num_layers = [len(self.shared_layers_size)-1] + [len(self.task_specific_layers_size[x]) for x in range(self.num_tasks)]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size
        if type(input_size) == list:
            tmp_size = 1
            for elem in input_size:
                tmp_size *= elem
            self.input_size = tmp_size
        else:
            self.input_size = input_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)

        if saveData:
            self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)
            self.data_index = tf.placeholder(dtype=tf.int32)
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data(self.data, self.batch_size, self.data_index, do_MTL=True)
        else:
            self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size]) for _ in range(self.num_tasks)]
            self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data_not_in_gpu(self.model_input, self.true_output, do_MTL=True)

        #### layers of model for train data
        self.train_models, self.shared_param, self.specific_param = new_tensorfactored_fc_fc_nets(train_x_batch, self.shared_layers_size, self.task_specific_layers_size, self.num_tasks, activation_fn=tf.nn.relu, shared_params=None, specific_params=None, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k, output_type='classification')
        self.param = self.shared_param + self.specific_param

        #### layers of model for validation data
        self.valid_models, _, _ = new_tensorfactored_fc_fc_nets(valid_x_batch, self.shared_layers_size, self.task_specific_layers_size, self.num_tasks, activation_fn=tf.nn.relu, shared_params=self.shared_param, specific_params=self.specific_param, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k, output_type='classification')

        #### layers of model for test data
        self.test_models, _, _ = new_tensorfactored_fc_fc_nets(test_x_batch, self.shared_layers_size, self.task_specific_layers_size, self.num_tasks, activation_fn=tf.nn.relu, shared_params=self.shared_param, specific_params=self.specific_param, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k, output_type='classification')

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.task_specific_layers_size[0][-1], classification)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()
