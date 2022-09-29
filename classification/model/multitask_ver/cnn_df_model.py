import numpy as np
import tensorflow as tf

from utils.utils import *
from utils.utils_df_nn import *


#### if linear relation, set 'relation_activation_fn' None
#### if nonlinear relation, set 'relation_activation_fn' with activation function such as tf.nn.relu
class ELLA_CNN_relation2_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernels, dim_strides, dim_cnn_know_base, dim_fc_know_base, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, relation_activation_fn=None, saveData=False, skip_connect=[]):
        self.num_tasks = num_tasks
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.num_cnn_KB_para = len(dim_channels)
        self.num_cnn_TS_para = 0
        self.num_fc_KB_para = len(dim_fcs)
        self.num_fc_TS_para = 0
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernels     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.cnn_KB_size = dim_cnn_know_base
        self.fc_KB_size = dim_fc_know_base
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.KB_l1_reg_scale = reg_scale[0]
        self.KB_l2_reg_scale = reg_scale[1]
        self.TS_l1_reg_scale = reg_scale[2]
        self.TS_l2_reg_scale = reg_scale[3]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        if saveData:
            self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)
            self.data_index = tf.placeholder(dtype=tf.int32)
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data(self.data, self.batch_size, self.data_index, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)
        else:
            self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]]) for _ in range(self.num_tasks)]
            self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data_not_in_gpu(self.model_input, self.true_output, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)

        ### define regularizer
        KB_l1_reg, KB_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.KB_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.KB_l2_reg_scale)
        TS_l1_reg, TS_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.TS_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.TS_l2_reg_scale)

        #### layers of model for train data
        self.train_models = []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.cnn_KB_param, self.cnn_TS_param, self.fc_KB_param, self.fc_TS_param = new_ELLA_cnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.fc_KB_size, cnn_para_activation_fn=relation_activation_fn, cnn_KB_params=None, cnn_TS_params=None, fc_para_activation_fn=relation_activation_fn, fc_KB_params=None, fc_TS_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
                self.num_cnn_TS_para = len(self.cnn_TS_param)
                self.num_fc_TS_para = len(self.fc_TS_param)
            else:
                model_tmp, _, cnn_TS_param_tmp, _, fc_TS_param_tmp = new_ELLA_cnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.fc_KB_size, cnn_para_activation_fn=relation_activation_fn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=None, fc_para_activation_fn=relation_activation_fn, fc_KB_params=self.fc_KB_param, fc_TS_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
                self.cnn_TS_param = self.cnn_TS_param + cnn_TS_param_tmp
                self.fc_TS_param = self.fc_TS_param + fc_TS_param_tmp
            self.train_models.append(model_tmp)
        self.param = self.cnn_KB_param + self.cnn_TS_param + self.fc_KB_param + self.fc_TS_param

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.fc_KB_size, cnn_para_activation_fn=relation_activation_fn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt*self.num_cnn_TS_para:(task_cnt+1)*self.num_cnn_TS_para], fc_para_activation_fn=relation_activation_fn, fc_KB_params=self.fc_KB_param, fc_TS_params=self.fc_TS_param[task_cnt*self.num_fc_TS_para:(task_cnt+1)*self.num_fc_TS_para], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.fc_KB_size, cnn_para_activation_fn=relation_activation_fn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt*self.num_cnn_TS_para:(task_cnt+1)*self.num_cnn_TS_para], fc_para_activation_fn=relation_activation_fn, fc_KB_params=self.fc_KB_param, fc_TS_params=self.fc_TS_param[task_cnt*self.num_fc_TS_para:(task_cnt+1)*self.num_fc_TS_para], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.test_models.append(model_tmp)

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        KB_reg_term1, KB_reg_term2 = tf.contrib.layers.apply_regularization(KB_l1_reg, reg_var), tf.contrib.layers.apply_regularization(KB_l2_reg, reg_var)
        TS_reg_term1, TS_reg_term2 = tf.contrib.layers.apply_regularization(TS_l1_reg, reg_var), tf.contrib.layers.apply_regularization(TS_l2_reg, reg_var)

        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.fc_size[-1], classification=True)

        self.train_loss_and_reg = [self.train_loss[x] + KB_reg_term1 + KB_reg_term2+ TS_reg_term1 + TS_reg_term2 for x in range(self.num_tasks)]

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()


#### Used Deconvolution layer from KB to TS param
#### TS.W = para_activation_fn(Deconv(KB)+bias) // TS.bias
class MTL_DFCNN_direct_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernels, dim_strides, dim_cnn_know_base, dim_cnn_task_specific, dim_cnn_deconv_strides, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, relation_activation_fn_cnn=tf.nn.relu, saveData=False, skip_connect=[]):
        self.num_tasks = num_tasks
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.num_cnn_KB_para = len(dim_channels)
        self.num_cnn_TS_para = 0
        self.num_fc_para = 0
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernels     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.cnn_KB_size = dim_cnn_know_base
        self.cnn_TS_size = dim_cnn_task_specific
        self.cnn_deconv_stride_size = dim_cnn_deconv_strides
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.KB_l1_reg_scale = reg_scale[0]
        self.KB_l2_reg_scale = reg_scale[1]
        self.TS_l1_reg_scale = reg_scale[2]
        self.TS_l2_reg_scale = reg_scale[3]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        if saveData:
            self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)
            self.data_index = tf.placeholder(dtype=tf.int32)
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data(self.data, self.batch_size, self.data_index, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)
        else:
            self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]]) for _ in range(self.num_tasks)]
            self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data_not_in_gpu(self.model_input, self.true_output, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)

        ### define regularizer
        KB_l1_reg, KB_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.KB_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.KB_l2_reg_scale)
        TS_l1_reg, TS_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.TS_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.TS_l2_reg_scale)

        #### layers of model for train data
        self.train_models, self.cnn_TS_param, self.cnn_gen_param, self.fc_param = [], [], [], []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.cnn_KB_param, cnn_TS_param_tmp, cnn_gen_param_tmp, fc_param_tmp = new_ELLA_cnn_deconv_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=None, cnn_TS_params=None, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
                self.num_cnn_TS_para = len(cnn_TS_param_tmp)
                self.num_fc_para = len(fc_param_tmp)
            else:
                model_tmp, _, cnn_TS_param_tmp, cnn_gen_param_tmp, fc_param_tmp = new_ELLA_cnn_deconv_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=None, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.cnn_TS_param.append(cnn_TS_param_tmp)
            self.cnn_gen_param.append(cnn_gen_param_tmp)
            self.fc_param.append(fc_param_tmp)
            self.train_models.append(model_tmp)
        self.param = self.cnn_KB_param + sum(self.cnn_TS_param, []) + sum(self.fc_param, [])

        assert (len(self.cnn_KB_param) == self.num_cnn_KB_para), "CNN KB size doesn't match"
        assert (sum([len(self.cnn_TS_param[x]) for x in range(len(self.cnn_TS_param))]) == self.num_tasks * self.num_cnn_TS_para), "CNN TS size doesn't match"

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.test_models.append(model_tmp)

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        KB_reg_term1, KB_reg_term2 = tf.contrib.layers.apply_regularization(KB_l1_reg, reg_var), tf.contrib.layers.apply_regularization(KB_l2_reg, reg_var)
        TS_reg_term1, TS_reg_term2 = tf.contrib.layers.apply_regularization(TS_l1_reg, reg_var), tf.contrib.layers.apply_regularization(TS_l2_reg, reg_var)

        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.fc_size[-1], classification=True)

        #self.train_loss_and_reg = [self.train_loss[x] + KB_reg_term1 + KB_reg_term2+ TS_reg_term1 + TS_reg_term2 for x in range(self.num_tasks)]
        #self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        KB_grads = [tf.gradients(self.train_loss[x] + KB_reg_term1 + KB_reg_term2, self.cnn_KB_param) for x in range(self.num_tasks)]
        KB_grads_vars = [[(grad/(1.0+2.0*self.epoch*self.learn_rate_decay), param) for grad, param in zip(KB_grads[x], self.cnn_KB_param)] for x in range(self.num_tasks)]

        TS_grads = [tf.gradients(self.train_loss[x] + TS_reg_term1 + TS_reg_term2, self.cnn_TS_param[x]) for x in range(self.num_tasks)]
        TS_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(TS_grads[x], self.cnn_TS_param[x])] for x in range(self.num_tasks)]

        fc_grads = [tf.gradients(self.train_loss[x], self.fc_param[x]) for x in range(self.num_tasks)]
        fc_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(fc_grads[x], self.fc_param[x])] for x in range(self.num_tasks)]

        grads_vars = [KB_grads_vars[x] + TS_grads_vars[x] + fc_grads_vars[x] for x in range(self.num_tasks)]
        trainer = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate) for x in range(self.num_tasks)]
        self.update = [trainer[x].apply_gradients(grads_vars[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        KB_param_val = get_value_of_valid_tensors(sess, self.cnn_KB_param)
        TS_param_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.cnn_TS_param]
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.cnn_gen_param]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_param]

        parameters_val = {}
        parameters_val['conv_KB'] = savemat_wrapper(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val


#### Used Deconvolution layer from KB to TS param
#### TS.W = para_activation_fn(Deconv(KB)+bias) // TS.bias
class MTL_DFCNN_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernels, dim_strides, dim_cnn_know_base, dim_cnn_task_specific, dim_cnn_deconv_strides, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, relation_activation_fn_cnn=tf.nn.relu, saveData=False, skip_connect=[]):
        self.num_tasks = num_tasks
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.num_cnn_KB_para = len(dim_channels)
        self.num_cnn_TS_para = 0
        self.num_fc_para = 0
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernels     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.cnn_KB_size = dim_cnn_know_base
        self.cnn_TS_size = dim_cnn_task_specific
        self.cnn_deconv_stride_size = dim_cnn_deconv_strides
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.KB_l1_reg_scale = reg_scale[0]
        self.KB_l2_reg_scale = reg_scale[1]
        self.TS_l1_reg_scale = reg_scale[2]
        self.TS_l2_reg_scale = reg_scale[3]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        if saveData:
            self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)
            self.data_index = tf.placeholder(dtype=tf.int32)
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data(self.data, self.batch_size, self.data_index, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)
        else:
            self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]]) for _ in range(self.num_tasks)]
            self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data_not_in_gpu(self.model_input, self.true_output, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)

        ### define regularizer
        KB_l1_reg, KB_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.KB_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.KB_l2_reg_scale)
        TS_l1_reg, TS_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.TS_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.TS_l2_reg_scale)

        #### layers of model for train data
        self.train_models, self.cnn_TS_param, self.cnn_gen_param, self.fc_param = [], [], [], []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.cnn_KB_param, cnn_TS_param_tmp, cnn_gen_param_tmp, fc_param_tmp = new_ELLA_cnn_deconv_tensordot_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=None, cnn_TS_params=None, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
                self.num_cnn_TS_para = len(cnn_TS_param_tmp)
                self.num_fc_para = len(fc_param_tmp)
            else:
                model_tmp, _, cnn_TS_param_tmp, cnn_gen_param_tmp, fc_param_tmp = new_ELLA_cnn_deconv_tensordot_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=None, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.cnn_TS_param.append(cnn_TS_param_tmp)
            self.cnn_gen_param.append(cnn_gen_param_tmp)
            self.fc_param.append(fc_param_tmp)
            self.train_models.append(model_tmp)
        self.param = self.cnn_KB_param + sum(self.cnn_TS_param, []) + sum(self.fc_param, [])

        assert (len(self.cnn_KB_param) == self.num_cnn_KB_para), "CNN KB size doesn't match"
        assert (sum([len(self.cnn_TS_param[x]) for x in range(len(self.cnn_TS_param))]) == self.num_tasks * self.num_cnn_TS_para), "CNN TS size doesn't match"

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_tensordot_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_tensordot_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.test_models.append(model_tmp)

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        KB_reg_term1, KB_reg_term2 = tf.contrib.layers.apply_regularization(KB_l1_reg, reg_var), tf.contrib.layers.apply_regularization(KB_l2_reg, reg_var)
        TS_reg_term1, TS_reg_term2 = tf.contrib.layers.apply_regularization(TS_l1_reg, reg_var), tf.contrib.layers.apply_regularization(TS_l2_reg, reg_var)

        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.fc_size[-1], classification=True)

        #self.train_loss_and_reg = [self.train_loss[x] + KB_reg_term1 + KB_reg_term2+ TS_reg_term1 + TS_reg_term2 for x in range(self.num_tasks)]

        #self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        KB_grads = [tf.gradients(self.train_loss[x] + KB_reg_term1 + KB_reg_term2, self.cnn_KB_param) for x in range(self.num_tasks)]
        KB_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(KB_grads[x], self.cnn_KB_param)] for x in range(self.num_tasks)]

        TS_grads = [tf.gradients(self.train_loss[x] + TS_reg_term1 + TS_reg_term2, self.cnn_TS_param[x]) for x in range(self.num_tasks)]
        TS_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(TS_grads[x], self.cnn_TS_param[x])] for x in range(self.num_tasks)]

        fc_grads = [tf.gradients(self.train_loss[x], self.fc_param[x]) for x in range(self.num_tasks)]
        fc_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(fc_grads[x], self.fc_param[x])] for x in range(self.num_tasks)]

        grads_vars = [KB_grads_vars[x] + TS_grads_vars[x] + fc_grads_vars[x] for x in range(self.num_tasks)]
        trainer = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate) for x in range(self.num_tasks)]
        self.update = [trainer[x].apply_gradients(grads_vars[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        KB_param_val = get_value_of_valid_tensors(sess, self.cnn_KB_param)
        TS_param_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.cnn_TS_param]
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.cnn_gen_param]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_param]

        parameters_val = {}
        parameters_val['conv_KB'] = savemat_wrapper(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val


#### Used Deconvolution layer from KB to TS param
#### TS.W = para_activation_fn(Deconv(KB)+bias) // TS.bias
class MTL_DFCNN_tc2_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernels, dim_strides, dim_cnn_know_base, dim_cnn_task_specific, dim_cnn_deconv_strides, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, relation_activation_fn_cnn=tf.nn.relu, saveData=False, skip_connect=[]):
        self.num_tasks = num_tasks
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.num_cnn_KB_para = len(dim_channels)
        self.num_cnn_TS_para = 0
        self.num_fc_para = 0
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernels     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.cnn_KB_size = dim_cnn_know_base
        self.cnn_TS_size = dim_cnn_task_specific
        self.cnn_deconv_stride_size = dim_cnn_deconv_strides
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.KB_l1_reg_scale = reg_scale[0]
        self.KB_l2_reg_scale = reg_scale[1]
        self.TS_l1_reg_scale = reg_scale[2]
        self.TS_l2_reg_scale = reg_scale[3]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        if saveData:
            self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)
            self.data_index = tf.placeholder(dtype=tf.int32)
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data(self.data, self.batch_size, self.data_index, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)
        else:
            self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]]) for _ in range(self.num_tasks)]
            self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data_not_in_gpu(self.model_input, self.true_output, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)

        ### define regularizer
        KB_l1_reg, KB_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.KB_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.KB_l2_reg_scale)
        TS_l1_reg, TS_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.TS_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.TS_l2_reg_scale)

        #### layers of model for train data
        self.train_models, self.cnn_TS_param, self.cnn_gen_param, self.fc_param = [], [], [], []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.cnn_KB_param, cnn_TS_param_tmp, cnn_gen_param_tmp, fc_param_tmp = new_ELLA_cnn_deconv_tensordot_fc_net2(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=None, cnn_TS_params=None, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
                self.num_cnn_TS_para = len(cnn_TS_param_tmp)
                self.num_fc_para = len(fc_param_tmp)
            else:
                model_tmp, _, cnn_TS_param_tmp, cnn_gen_param_tmp, fc_param_tmp = new_ELLA_cnn_deconv_tensordot_fc_net2(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=None, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.cnn_TS_param.append(cnn_TS_param_tmp)
            self.cnn_gen_param.append(cnn_gen_param_tmp)
            self.fc_param.append(fc_param_tmp)
            self.train_models.append(model_tmp)
        self.param = self.cnn_KB_param + sum(self.cnn_TS_param, []) + sum(self.fc_param, [])

        assert (len(self.cnn_KB_param) == self.num_cnn_KB_para), "CNN KB size doesn't match"
        assert (sum([len(self.cnn_TS_param[x]) for x in range(len(self.cnn_TS_param))]) == self.num_tasks * self.num_cnn_TS_para), "CNN TS size doesn't match"

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_tensordot_fc_net2(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_tensordot_fc_net2(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.test_models.append(model_tmp)

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        KB_reg_term1, KB_reg_term2 = tf.contrib.layers.apply_regularization(KB_l1_reg, reg_var), tf.contrib.layers.apply_regularization(KB_l2_reg, reg_var)
        TS_reg_term1, TS_reg_term2 = tf.contrib.layers.apply_regularization(TS_l1_reg, reg_var), tf.contrib.layers.apply_regularization(TS_l2_reg, reg_var)

        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.fc_size[-1], classification=True)

        #self.train_loss_and_reg = [self.train_loss[x] + KB_reg_term1 + KB_reg_term2+ TS_reg_term1 + TS_reg_term2 for x in range(self.num_tasks)]

        #self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        KB_grads = [tf.gradients(self.train_loss[x] + KB_reg_term1 + KB_reg_term2, self.cnn_KB_param) for x in range(self.num_tasks)]
        KB_grads_vars = [[(grad/(1.0+2.0*self.epoch*self.learn_rate_decay), param) for grad, param in zip(KB_grads[x], self.cnn_KB_param)] for x in range(self.num_tasks)]

        TS_grads = [tf.gradients(self.train_loss[x] + TS_reg_term1 + TS_reg_term2, self.cnn_TS_param[x]) for x in range(self.num_tasks)]
        TS_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(TS_grads[x], self.cnn_TS_param[x])] for x in range(self.num_tasks)]

        fc_grads = [tf.gradients(self.train_loss[x], self.fc_param[x]) for x in range(self.num_tasks)]
        fc_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(fc_grads[x], self.fc_param[x])] for x in range(self.num_tasks)]

        grads_vars = [KB_grads_vars[x] + TS_grads_vars[x] + fc_grads_vars[x] for x in range(self.num_tasks)]
        trainer = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate) for x in range(self.num_tasks)]
        self.update = [trainer[x].apply_gradients(grads_vars[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        KB_param_val = get_value_of_valid_tensors(sess, self.cnn_KB_param)
        TS_param_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.cnn_TS_param]
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.cnn_gen_param]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_param]

        parameters_val = {}
        parameters_val['conv_KB'] = savemat_wrapper(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val


#### Used Deconvolution layer from KB to TS param
#### TS.W = para_activation_fn(Deconv(KB)+bias) // TS.bias
class ELLA_CNN_deconv_tensordot_relation_minibatch2_reshape():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernels, dim_strides, dim_cnn_know_base, dim_cnn_task_specific, dim_cnn_deconv_strides, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, relation_activation_fn_cnn=tf.nn.relu, saveData=False, skip_connect=[]):
        self.num_tasks = num_tasks
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.num_cnn_KB_para = len(dim_channels)
        self.num_cnn_TS_para = 0
        self.num_fc_para = 0
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernels     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.cnn_KB_size = dim_cnn_know_base
        self.cnn_TS_size = dim_cnn_task_specific
        self.cnn_deconv_stride_size = dim_cnn_deconv_strides
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.KB_l1_reg_scale = reg_scale[0]
        self.KB_l2_reg_scale = reg_scale[1]
        self.TS_l1_reg_scale = reg_scale[2]
        self.TS_l2_reg_scale = reg_scale[3]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        if saveData:
            self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)
            self.data_index = tf.placeholder(dtype=tf.int32)
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data(self.data, self.batch_size, self.data_index, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)
        else:
            self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]]) for _ in range(self.num_tasks)]
            self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data_not_in_gpu(self.model_input, self.true_output, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)

        ### define regularizer
        KB_l1_reg, KB_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.KB_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.KB_l2_reg_scale)
        TS_l1_reg, TS_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.TS_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.TS_l2_reg_scale)

        #### layers of model for train data
        self.train_models, self.cnn_TS_param, self.cnn_gen_param, self.fc_param = [], [], [], []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.cnn_KB_param, cnn_TS_param_tmp, cnn_gen_param_tmp, fc_param_tmp = new_ELLA_cnn_deconv_tensordot_fc_net2_reshape(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=None, cnn_TS_params=None, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
                self.num_cnn_TS_para = len(cnn_TS_param_tmp)
                self.num_fc_para = len(fc_param_tmp)
            else:
                model_tmp, _, cnn_TS_param_tmp, cnn_gen_param_tmp, fc_param_tmp = new_ELLA_cnn_deconv_tensordot_fc_net2_reshape(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=None, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.cnn_TS_param.append(cnn_TS_param_tmp)
            self.cnn_gen_param.append(cnn_gen_param_tmp)
            self.fc_param.append(fc_param_tmp)
            self.train_models.append(model_tmp)
        self.param = self.cnn_KB_param + sum(self.cnn_TS_param, []) + sum(self.fc_param, [])

        assert (len(self.cnn_KB_param) == self.num_cnn_KB_para), "CNN KB size doesn't match"
        assert (sum([len(self.cnn_TS_param[x]) for x in range(len(self.cnn_TS_param))]) == self.num_tasks * self.num_cnn_TS_para), "CNN TS size doesn't match"

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_tensordot_fc_net2_reshape(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_tensordot_fc_net2_reshape(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.test_models.append(model_tmp)


        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        KB_reg_term1, KB_reg_term2 = tf.contrib.layers.apply_regularization(KB_l1_reg, reg_var), tf.contrib.layers.apply_regularization(KB_l2_reg, reg_var)
        TS_reg_term1, TS_reg_term2 = tf.contrib.layers.apply_regularization(TS_l1_reg, reg_var), tf.contrib.layers.apply_regularization(TS_l2_reg, reg_var)

        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.fc_size[-1], classification=True)

        #self.train_loss_and_reg = [self.train_loss[x] + KB_reg_term1 + KB_reg_term2+ TS_reg_term1 + TS_reg_term2 for x in range(self.num_tasks)]

        #self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        KB_grads = [tf.gradients(self.train_loss[x] + KB_reg_term1 + KB_reg_term2, self.cnn_KB_param) for x in range(self.num_tasks)]
        KB_grads_vars = [[(grad/(1.0+2.0*self.epoch*self.learn_rate_decay), param) for grad, param in zip(KB_grads[x], self.cnn_KB_param)] for x in range(self.num_tasks)]

        TS_grads = [tf.gradients(self.train_loss[x] + TS_reg_term1 + TS_reg_term2, self.cnn_TS_param[x]) for x in range(self.num_tasks)]
        TS_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(TS_grads[x], self.cnn_TS_param[x])] for x in range(self.num_tasks)]

        fc_grads = [tf.gradients(self.train_loss[x], self.fc_param[x]) for x in range(self.num_tasks)]
        fc_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(fc_grads[x], self.fc_param[x])] for x in range(self.num_tasks)]

        grads_vars = [KB_grads_vars[x] + TS_grads_vars[x] + fc_grads_vars[x] for x in range(self.num_tasks)]
        trainer = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate) for x in range(self.num_tasks)]
        self.update = [trainer[x].apply_gradients(grads_vars[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        KB_param_val = get_value_of_valid_tensors(sess, self.cnn_KB_param)
        TS_param_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.cnn_TS_param]
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.cnn_gen_param]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_param]

        parameters_val = {}
        parameters_val['conv_KB'] = savemat_wrapper(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val


#### Used Deconvolution layer from KB to TS param
#### TS.W = para_activation_fn(Deconv(KB)+bias) // TS.bias
class ELLA_CNN_deconv_tensordot_relation_reverse_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernels, dim_strides, dim_cnn_know_base, dim_cnn_task_specific, dim_cnn_deconv_strides, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, relation_activation_fn_cnn=tf.nn.relu, saveData=False, skip_connect=[]):
        self.num_tasks = num_tasks
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.num_cnn_KB_para = 0
        self.num_cnn_TS_para = 0
        self.num_fc_para = 0
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernels     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.cnn_KB_size = dim_cnn_know_base
        self.cnn_TS_size = dim_cnn_task_specific
        self.cnn_deconv_stride_size = dim_cnn_deconv_strides
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.KB_l1_reg_scale = reg_scale[0]
        self.KB_l2_reg_scale = reg_scale[1]
        self.TS_l1_reg_scale = reg_scale[2]
        self.TS_l2_reg_scale = reg_scale[3]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        if saveData:
            self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)
            self.data_index = tf.placeholder(dtype=tf.int32)
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data(self.data, self.batch_size, self.data_index, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)
        else:
            self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]]) for _ in range(self.num_tasks)]
            self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data_not_in_gpu(self.model_input, self.true_output, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)

        ### define regularizer
        KB_l1_reg, KB_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.KB_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.KB_l2_reg_scale)
        TS_l1_reg, TS_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.TS_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.TS_l2_reg_scale)

        #### layers of model for train data
        self.train_models, self.cnn_KB_param, self.cnn_gen_param, self.fc_param = [], [], [], []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, cnn_KB_param_tmp, self.cnn_TS_param, cnn_gen_param_tmp, fc_param_tmp = new_ELLA_cnn_deconv_tensordot_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=None, cnn_TS_params=None, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
                self.num_cnn_KB_para = len(cnn_KB_param_tmp)
                self.num_cnn_TS_para = len(self.cnn_TS_param)
                self.num_fc_para = len(fc_param_tmp)
            else:
                model_tmp, cnn_KB_param_tmp, _, cnn_gen_param_tmp, fc_param_tmp = new_ELLA_cnn_deconv_tensordot_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=None, cnn_TS_params=self.cnn_TS_param, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.cnn_KB_param.append(cnn_KB_param_tmp)
            self.cnn_gen_param.append(cnn_gen_param_tmp)
            self.fc_param.append(fc_param_tmp)
            self.train_models.append(model_tmp)
        self.param = sum(self.cnn_KB_param, []) + self.cnn_TS_param + sum(self.fc_param, [])

        assert (sum([len(self.cnn_KB_param[x]) for x in range(len(self.cnn_KB_param))]) == self.num_tasks * self.num_cnn_KB_para), "CNN KB size doesn't match"
        assert (len(self.cnn_TS_param) == self.num_cnn_TS_para), "CNN TS size doesn't match"

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_tensordot_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param[task_cnt], cnn_TS_params=self.cnn_TS_param, fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_tensordot_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param[task_cnt], cnn_TS_params=self.cnn_TS_param, fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.test_models.append(model_tmp)


        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        KB_reg_term1, KB_reg_term2 = tf.contrib.layers.apply_regularization(KB_l1_reg, reg_var), tf.contrib.layers.apply_regularization(KB_l2_reg, reg_var)
        TS_reg_term1, TS_reg_term2 = tf.contrib.layers.apply_regularization(TS_l1_reg, reg_var), tf.contrib.layers.apply_regularization(TS_l2_reg, reg_var)

        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.fc_size[-1], classification=True)

        #self.train_loss_and_reg = [self.train_loss[x] + KB_reg_term1 + KB_reg_term2+ TS_reg_term1 + TS_reg_term2 for x in range(self.num_tasks)]

        #self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        KB_grads = [tf.gradients(self.train_loss[x] + KB_reg_term1 + KB_reg_term2, self.cnn_KB_param[x]) for x in range(self.num_tasks)]
        KB_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(KB_grads[x], self.cnn_KB_param[x])] for x in range(self.num_tasks)]

        TS_grads = [tf.gradients(self.train_loss[x] + TS_reg_term1 + TS_reg_term2, self.cnn_TS_param) for x in range(self.num_tasks)]
        TS_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(TS_grads[x], self.cnn_TS_param)] for x in range(self.num_tasks)]

        fc_grads = [tf.gradients(self.train_loss[x], self.fc_param[x]) for x in range(self.num_tasks)]
        fc_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(fc_grads[x], self.fc_param[x])] for x in range(self.num_tasks)]

        grads_vars = [KB_grads_vars[x] + TS_grads_vars[x] + fc_grads_vars[x] for x in range(self.num_tasks)]
        trainer = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate) for x in range(self.num_tasks)]
        self.update = [trainer[x].apply_gradients(grads_vars[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        KB_param_val = [get_value_of_valid_tensors(sess, cnn_KB_param) for cnn_KB_param in self.cnn_KB_param]
        TS_param_val = get_value_of_valid_tensors(sess, self.cnn_TS_param)
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.cnn_gen_param]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_param]

        parameters_val = {}
        parameters_val['conv_KB'] = savemat_wrapper_nested_list(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val


#### Used Deconvolution layer from KB to TS param
#### TS.W = para_activation_fn(Deconv(KB)+bias) // TS.bias
class ELLA_CNN_deconv_tensordot_relation_reverse_minibatch2():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernels, dim_strides, dim_cnn_know_base, dim_cnn_task_specific, dim_cnn_deconv_strides, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, relation_activation_fn_cnn=tf.nn.relu, saveData=False, skip_connect=[]):
        self.num_tasks = num_tasks
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.num_cnn_KB_para = 0
        self.num_cnn_TS_para = 0
        self.num_fc_para = 0
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernels     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.cnn_KB_size = dim_cnn_know_base
        self.cnn_TS_size = dim_cnn_task_specific
        self.cnn_deconv_stride_size = dim_cnn_deconv_strides
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.KB_l1_reg_scale = reg_scale[0]
        self.KB_l2_reg_scale = reg_scale[1]
        self.TS_l1_reg_scale = reg_scale[2]
        self.TS_l2_reg_scale = reg_scale[3]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        if saveData:
            self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)
            self.data_index = tf.placeholder(dtype=tf.int32)
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data(self.data, self.batch_size, self.data_index, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)
        else:
            self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]]) for _ in range(self.num_tasks)]
            self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data_not_in_gpu(self.model_input, self.true_output, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)

        ### define regularizer
        KB_l1_reg, KB_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.KB_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.KB_l2_reg_scale)
        TS_l1_reg, TS_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.TS_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.TS_l2_reg_scale)

        #### layers of model for train data
        self.train_models, self.cnn_KB_param, self.cnn_gen_param, self.fc_param = [], [], [], []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, cnn_KB_param_tmp, self.cnn_TS_param, cnn_gen_param_tmp, fc_param_tmp = new_ELLA_cnn_deconv_tensordot_fc_net2(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=None, cnn_TS_params=None, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
                self.num_cnn_KB_para = len(cnn_KB_param_tmp)
                self.num_cnn_TS_para = len(self.cnn_TS_param)
                self.num_fc_para = len(fc_param_tmp)
            else:
                model_tmp, cnn_KB_param_tmp, _, cnn_gen_param_tmp, fc_param_tmp = new_ELLA_cnn_deconv_tensordot_fc_net2(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=None, cnn_TS_params=self.cnn_TS_param, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.cnn_KB_param.append(cnn_KB_param_tmp)
            self.cnn_gen_param.append(cnn_gen_param_tmp)
            self.fc_param.append(fc_param_tmp)
            self.train_models.append(model_tmp)
        self.param = sum(self.cnn_KB_param, []) + self.cnn_TS_param + sum(self.fc_param, [])

        assert (sum([len(self.cnn_KB_param[x]) for x in range(len(self.cnn_KB_param))]) == self.num_tasks * self.num_cnn_KB_para), "CNN KB size doesn't match"
        assert (len(self.cnn_TS_param) == self.num_cnn_TS_para), "CNN TS size doesn't match"

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_tensordot_fc_net2(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param[task_cnt], cnn_TS_params=self.cnn_TS_param, fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_tensordot_fc_net2(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param[task_cnt], cnn_TS_params=self.cnn_TS_param, fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.test_models.append(model_tmp)


        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        KB_reg_term1, KB_reg_term2 = tf.contrib.layers.apply_regularization(KB_l1_reg, reg_var), tf.contrib.layers.apply_regularization(KB_l2_reg, reg_var)
        TS_reg_term1, TS_reg_term2 = tf.contrib.layers.apply_regularization(TS_l1_reg, reg_var), tf.contrib.layers.apply_regularization(TS_l2_reg, reg_var)

        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.fc_size[-1], classification=True)

        #self.train_loss_and_reg = [self.train_loss[x] + KB_reg_term1 + KB_reg_term2+ TS_reg_term1 + TS_reg_term2 for x in range(self.num_tasks)]

        #self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        KB_grads = [tf.gradients(self.train_loss[x] + KB_reg_term1 + KB_reg_term2, self.cnn_KB_param[x]) for x in range(self.num_tasks)]
        KB_grads_vars = [[(grad/(1.0+2.0*self.epoch*self.learn_rate_decay), param) for grad, param in zip(KB_grads[x], self.cnn_KB_param[x])] for x in range(self.num_tasks)]

        TS_grads = [tf.gradients(self.train_loss[x] + TS_reg_term1 + TS_reg_term2, self.cnn_TS_param) for x in range(self.num_tasks)]
        TS_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(TS_grads[x], self.cnn_TS_param)] for x in range(self.num_tasks)]

        fc_grads = [tf.gradients(self.train_loss[x], self.fc_param[x]) for x in range(self.num_tasks)]
        fc_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(fc_grads[x], self.fc_param[x])] for x in range(self.num_tasks)]

        grads_vars = [KB_grads_vars[x] + TS_grads_vars[x] + fc_grads_vars[x] for x in range(self.num_tasks)]
        trainer = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate) for x in range(self.num_tasks)]
        self.update = [trainer[x].apply_gradients(grads_vars[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        KB_param_val = [get_value_of_valid_tensors(sess, cnn_KB_param) for cnn_KB_param in self.cnn_KB_param]
        TS_param_val = get_value_of_valid_tensors(sess, self.cnn_TS_param)
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.cnn_gen_param]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_param]

        parameters_val = {}
        parameters_val['conv_KB'] = savemat_wrapper_nested_list(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val




#### Used Deconvolution layer from KB to TS param
#### TS.W = para_activation_fn(Deconv(KB)+bias) // TS.bias
class MTL_HybridDFCNN_resnet_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernels, dim_strides, cnn_sharing, dim_cnn_know_base, dim_cnn_task_specific, dim_cnn_deconv_strides, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, relation_activation_fn_cnn=tf.nn.relu, saveData=False, skip_connect=[]):
        self.num_tasks = num_tasks
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        #self.num_layers = len(dim_channels) + len(dim_fcs)
        self.num_cnn_KB_para = len(dim_channels)
        self.num_cnn_TS_para = 0
        self.num_fc_para = 0
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernels     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.cnn_sharing = cnn_sharing
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.cnn_KB_size = dim_cnn_know_base
        self.cnn_TS_size = dim_cnn_task_specific
        self.cnn_deconv_stride_size = dim_cnn_deconv_strides
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.KB_l1_reg_scale = reg_scale[0]
        self.KB_l2_reg_scale = reg_scale[1]
        self.TS_l1_reg_scale = reg_scale[2]
        self.TS_l2_reg_scale = reg_scale[3]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        if saveData:
            self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)
            self.data_index = tf.placeholder(dtype=tf.int32)
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data(self.data, self.batch_size, self.data_index, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)
        else:
            self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]]) for _ in range(self.num_tasks)]
            self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data_not_in_gpu(self.model_input, self.true_output, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)

        ### define regularizer
        KB_l1_reg, KB_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.KB_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.KB_l2_reg_scale)
        TS_l1_reg, TS_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.TS_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.TS_l2_reg_scale)

        #### layers of model for train data
        self.train_models, self.cnn_TS_param, self.cnn_gen_param, self.cnn_param, self.fc_param = [], [], [], [], []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.cnn_KB_param, cnn_TS_param_tmp, cnn_gen_param_tmp, cnn_param_tmp, _, fc_param_tmp = new_ELLA_flexible_cnn_deconv_tensordot_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size[task_cnt], self.cnn_sharing, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=None, cnn_TS_params=None, cnn_params=None, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
                self.num_cnn_TS_para = len(cnn_TS_param_tmp)
                self.num_fc_para = len(fc_param_tmp)
            else:
                model_tmp, _, cnn_TS_param_tmp, cnn_gen_param_tmp, cnn_param_tmp, _, fc_param_tmp = new_ELLA_flexible_cnn_deconv_tensordot_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size[task_cnt], self.cnn_sharing, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=None, cnn_params=None, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.cnn_TS_param.append(cnn_TS_param_tmp)
            self.cnn_gen_param.append(cnn_gen_param_tmp)
            self.cnn_param.append(cnn_param_tmp)
            self.fc_param.append(fc_param_tmp)
            self.train_models.append(model_tmp)
        self.param = self.cnn_KB_param + sum(self.cnn_TS_param, []) + sum(self.cnn_param, []) + sum(self.fc_param, [])
        self.cnn_KB_trainable_param = get_list_of_valid_tensors(self.cnn_KB_param)
        self.cnn_TS_trainable_param = [get_list_of_valid_tensors(cnn_ts_p) for cnn_ts_p in self.cnn_TS_param]
        self.cnn_trainable_param = [get_list_of_valid_tensors(cnn_p) for cnn_p in self.cnn_param]
        self.fc_trainable_param = [get_list_of_valid_tensors(fc_p) for fc_p in self.fc_param]

        assert (len(self.cnn_KB_param) == self.num_cnn_KB_para), "CNN KB size doesn't match"
        assert (sum([len(self.cnn_TS_param[x]) for x in range(len(self.cnn_TS_param))]) == self.num_tasks * self.num_cnn_TS_para), "CNN TS size doesn't match"

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _, _, _ = new_ELLA_flexible_cnn_deconv_tensordot_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size[task_cnt], self.cnn_sharing, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], cnn_params=self.cnn_param[task_cnt], fc_params=self.fc_param[task_cnt], KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _, _, _ = new_ELLA_flexible_cnn_deconv_tensordot_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size[task_cnt], self.cnn_sharing, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], cnn_params=self.cnn_param[task_cnt], fc_params=self.fc_param[task_cnt], KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.test_models.append(model_tmp)


        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        KB_reg_term1, KB_reg_term2 = tf.contrib.layers.apply_regularization(KB_l1_reg, reg_var), tf.contrib.layers.apply_regularization(KB_l2_reg, reg_var)
        TS_reg_term1, TS_reg_term2 = tf.contrib.layers.apply_regularization(TS_l1_reg, reg_var), tf.contrib.layers.apply_regularization(TS_l2_reg, reg_var)

        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.fc_size[-1], classification=True)

        #self.train_loss_and_reg = [self.train_loss[x] + KB_reg_term1 + KB_reg_term2+ TS_reg_term1 + TS_reg_term2 for x in range(self.num_tasks)]

        #self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        KB_grads = [tf.gradients(self.train_loss[x] + KB_reg_term1 + KB_reg_term2, self.cnn_KB_trainable_param) for x in range(self.num_tasks)]
        KB_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(KB_grads[x], self.cnn_KB_trainable_param)] for x in range(self.num_tasks)]

        TS_grads = [tf.gradients(self.train_loss[x] + TS_reg_term1 + TS_reg_term2, self.cnn_TS_trainable_param[x]) for x in range(self.num_tasks)]
        TS_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(TS_grads[x], self.cnn_TS_trainable_param[x])] for x in range(self.num_tasks)]

        conv_grads = [tf.gradients(self.train_loss[x], self.cnn_trainable_param[x]) for x in range(self.num_tasks)]
        conv_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(conv_grads[x], self.cnn_trainable_param[x])] for x in range(self.num_tasks)]

        fc_grads = [tf.gradients(self.train_loss[x], self.fc_trainable_param[x]) for x in range(self.num_tasks)]
        fc_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(fc_grads[x], self.fc_trainable_param[x])] for x in range(self.num_tasks)]

        grads_vars = [KB_grads_vars[x] + TS_grads_vars[x] + conv_grads_vars[x] + fc_grads_vars[x] for x in range(self.num_tasks)]
        trainer = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate) for x in range(self.num_tasks)]
        self.update = [trainer[x].apply_gradients(grads_vars[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        KB_param_val = get_value_of_valid_tensors(sess, self.cnn_KB_param)
        TS_param_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.cnn_TS_param]
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.cnn_gen_param]
        cnn_param_val = [get_value_of_valid_tensors(sess, cnn_param) for cnn_param in self.cnn_param]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_param]

        parameters_val = {}
        parameters_val['conv_KB'] = savemat_wrapper(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['conv_trainable_weights'] = savemat_wrapper_nested_list(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val







#### Used Deconvolution layer from KB to TS param
#### TS.W = para_activation_fn(Deconv(KB)+bias) // TS.bias
class MTL_HybridDFCNN_highway_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernels, dim_strides, cnn_sharing, dim_cnn_know_base, dim_cnn_task_specific, dim_cnn_deconv_strides, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, relation_activation_fn_cnn=tf.nn.relu, saveData=False, highway_connect_type=0):
        self.num_tasks = num_tasks
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.num_cnn_KB_para = len(dim_channels)
        self.num_cnn_TS_para = 0
        self.num_fc_para = 0
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernels     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.cnn_sharing = cnn_sharing
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.cnn_KB_size = dim_cnn_know_base
        self.cnn_TS_size = dim_cnn_task_specific
        self.cnn_deconv_stride_size = dim_cnn_deconv_strides
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.KB_l1_reg_scale = reg_scale[0]
        self.KB_l2_reg_scale = reg_scale[1]
        self.TS_l1_reg_scale = reg_scale[2]
        self.TS_l2_reg_scale = reg_scale[3]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        if saveData:
            self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)
            self.data_index = tf.placeholder(dtype=tf.int32)
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data(self.data, self.batch_size, self.data_index, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)
        else:
            self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]]) for _ in range(self.num_tasks)]
            self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data_not_in_gpu(self.model_input, self.true_output, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)

        ### define regularizer
        KB_l1_reg, KB_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.KB_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.KB_l2_reg_scale)
        TS_l1_reg, TS_l2_reg = tf.contrib.layers.l1_regularizer(scale=self.TS_l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.TS_l2_reg_scale)

        #### layers of model for train data
        self.train_models, self.cnn_TS_param, self.cnn_gen_param, self.cnn_param, self.fc_param, self.cnn_highway_param = [], [], [], [], [], []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.cnn_KB_param, cnn_TS_param_tmp, cnn_gen_param_tmp, cnn_param_tmp, cnn_highway_param_tmp, fc_param_tmp = new_ELLA_flexible_cnn_deconv_tensordot_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_sharing, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=None, cnn_TS_params=None, cnn_params=None, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, highway_connect_type=highway_connect_type, cnn_highway_params=None)
                self.num_cnn_TS_para = len(cnn_TS_param_tmp)
                self.num_fc_para = len(fc_param_tmp)
            else:
                model_tmp, _, cnn_TS_param_tmp, cnn_gen_param_tmp, cnn_param_tmp, cnn_highway_param_tmp, fc_param_tmp = new_ELLA_flexible_cnn_deconv_tensordot_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_sharing, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=None, cnn_params=None, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, highway_connect_type=highway_connect_type, cnn_highway_params=None)
            self.cnn_TS_param.append(cnn_TS_param_tmp)
            self.cnn_gen_param.append(cnn_gen_param_tmp)
            self.cnn_param.append(cnn_param_tmp)
            self.cnn_highway_param.append(cnn_highway_param_tmp)
            self.fc_param.append(fc_param_tmp)
            self.train_models.append(model_tmp)
        self.param = self.cnn_KB_param + sum(self.cnn_TS_param, []) + sum(self.cnn_param, []) + sum(self.fc_param, [])
        self.cnn_KB_trainable_param = get_list_of_valid_tensors(self.cnn_KB_param)
        self.cnn_TS_trainable_param = [get_list_of_valid_tensors(cnn_ts_p) for cnn_ts_p in self.cnn_TS_param]
        self.cnn_trainable_param = [get_list_of_valid_tensors(cnn_p) for cnn_p in self.cnn_param]
        self.cnn_highway_trainable_param = [get_list_of_valid_tensors(cnn_h) for cnn_h in self.cnn_highway_param]
        self.fc_trainable_param = [get_list_of_valid_tensors(fc_p) for fc_p in self.fc_param]

        assert (len(self.cnn_KB_param) == self.num_cnn_KB_para), "CNN KB size doesn't match"
        assert (sum([len(self.cnn_TS_param[x]) for x in range(len(self.cnn_TS_param))]) == self.num_tasks * self.num_cnn_TS_para), "CNN TS size doesn't match"

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _, _, _ = new_ELLA_flexible_cnn_deconv_tensordot_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_sharing, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], cnn_params=self.cnn_param[task_cnt], fc_params=self.fc_param[task_cnt], KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, highway_connect_type=highway_connect_type, cnn_highway_params=self.cnn_highway_param[task_cnt])
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _, _, _ = new_ELLA_flexible_cnn_deconv_tensordot_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_sharing, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param, cnn_TS_params=self.cnn_TS_param[task_cnt], cnn_params=self.cnn_param[task_cnt], fc_params=self.fc_param[task_cnt], KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, highway_connect_type=highway_connect_type, cnn_highway_params=self.cnn_highway_param[task_cnt])
            self.test_models.append(model_tmp)


        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        KB_reg_term1, KB_reg_term2 = tf.contrib.layers.apply_regularization(KB_l1_reg, reg_var), tf.contrib.layers.apply_regularization(KB_l2_reg, reg_var)
        TS_reg_term1, TS_reg_term2 = tf.contrib.layers.apply_regularization(TS_l1_reg, reg_var), tf.contrib.layers.apply_regularization(TS_l2_reg, reg_var)

        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.fc_size[-1], classification=True)

        #self.train_loss_and_reg = [self.train_loss[x] + KB_reg_term1 + KB_reg_term2+ TS_reg_term1 + TS_reg_term2 for x in range(self.num_tasks)]

        #self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        KB_grads = [tf.gradients(self.train_loss[x] + KB_reg_term1 + KB_reg_term2, self.cnn_KB_trainable_param) for x in range(self.num_tasks)]
        KB_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(KB_grads[x], self.cnn_KB_trainable_param)] for x in range(self.num_tasks)]

        TS_grads = [tf.gradients(self.train_loss[x] + TS_reg_term1 + TS_reg_term2, self.cnn_TS_trainable_param[x]) for x in range(self.num_tasks)]
        TS_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(TS_grads[x], self.cnn_TS_trainable_param[x])] for x in range(self.num_tasks)]

        conv_grads = [tf.gradients(self.train_loss[x], self.cnn_trainable_param[x]) for x in range(self.num_tasks)]
        conv_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(conv_grads[x], self.cnn_trainable_param[x])] for x in range(self.num_tasks)]

        conv_highway_grads = [tf.gradients(self.train_loss[x], self.cnn_highway_trainable_param[x]) for x in range(self.num_tasks)]
        conv_highway_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(conv_highway_grads[x], self.cnn_highway_trainable_param[x])] for x in range(self.num_tasks)]

        fc_grads = [tf.gradients(self.train_loss[x], self.fc_trainable_param[x]) for x in range(self.num_tasks)]
        fc_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(fc_grads[x], self.fc_trainable_param[x])] for x in range(self.num_tasks)]

        grads_vars = [KB_grads_vars[x] + TS_grads_vars[x] + conv_grads_vars[x] + conv_highway_grads_vars[x] + fc_grads_vars[x] for x in range(self.num_tasks)]
        trainer = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate) for x in range(self.num_tasks)]
        self.update = [trainer[x].apply_gradients(grads_vars[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        KB_param_val = get_value_of_valid_tensors(sess, self.cnn_KB_param)
        TS_param_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.cnn_TS_param]
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.cnn_gen_param]
        cnn_param_val = [get_value_of_valid_tensors(sess, cnn_param) for cnn_param in self.cnn_param]
        cnn_highway_param_val = [get_value_of_valid_tensors(sess, cnn_highway_param) for cnn_highway_param in self.cnn_highway_param]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_param]

        parameters_val = {}
        parameters_val['conv_KB'] = savemat_wrapper(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['conv_trainable_weights'] = savemat_wrapper_nested_list(cnn_param_val)
        parameters_val['conv_highway_weights'] = savemat_wrapper_nested_list(cnn_highway_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val