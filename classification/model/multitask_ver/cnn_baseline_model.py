import numpy as np
import tensorflow as tf

from utils.utils import *
from utils.utils_nn import *
from utils.utils_df_nn import new_ELLA_cnn_deconv_tensordot_fc_net, new_ELLA_cnn_deconv_tensordot_fc_net2

from utils.utils_obsolete import new_tensorfactored_cnn_fc_nets

#################################################
#########       Simple CNN batch       ##########
#################################################
#### Convolutional & Fully-connected Neural Net
class CNN_batch():
    def __init__(self, dim_channels, dim_fcs, input_size, dim_kernel, dim_strides, learning_rate, learning_rate_decay, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False):
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*len(channels_size-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*len(channels_size-1)
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay

        #### placeholder of model
        self.model_input = new_placeholder([None, self.input_size[0]*self.input_size[1]*self.input_size[2]])
        self.true_output = new_placeholder([None, self.fc_size[-1]])
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        reshaped_input = tf.reshape(self.model_input, [-1]+self.input_size)

        #### layers of model
        self.layers, self.cnn_param, self.fc_param = new_cnn_fc_net(reshaped_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification')
        self.param = self.cnn_param + self.fc_param

        #### functions of model
        if self.fc_size[-1]>1:
            self.eval = tf.nn.softmax(self.layers[-1])
            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.true_output, logits=self.layers[-1])
            self.accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.eval, 1), tf.argmax(self.true_output, 1)), tf.float32))
        else:
            self.eval = self.layers[-1]
            self.loss = 2.0* tf.nn.l2_loss(self.eval-self.true_output)
            self.accuracy = tf.reduce_sum(tf.cast(tf.equal((self.eval>0.5), (self.true_output>0.5)), tf.float32))

        #self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.loss)
        self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.loss)

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        cnn_param_val = get_value_of_valid_tensors(sess, self.cnn_param)
        fc_param_val = get_value_of_valid_tensors(sess, self.fc_param)

        parameters_val = {}
        parameters_val['conv_trainable_weights'] = savemat_wrapper(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper(fc_param_val)
        return parameters_val


#################################################
#######      Simple CNN mini-batch       ########
#################################################
#### Convolutional & Fully-connected Neural Net
class CNN_minibatch():
    def __init__(self, dim_channels, dim_fcs, input_size, dim_kernel, dim_strides, batch_size, learning_rate, learning_rate_decay, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, saveData=False):
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        if saveData:
            self.data = tflized_data(data_list, do_MTL=False)
            self.data_index = tf.placeholder(dtype=tf.int32)
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data(self.data, self.batch_size, self.data_index, [-1] + self.input_size, do_MTL=False)
        else:
            self.model_input = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]])
            self.true_output = tf.placeholder(dtype=tf.float32, shape=[self.batch_size])
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data_not_in_gpu(self.model_input, self.true_output, [-1] + self.input_size, do_MTL=False)

        #### layers of model for train data
        self.train_layers, self.cnn_param, self.fc_param = new_cnn_fc_net(train_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification')
        self.param = self.cnn_param + self.fc_param

        #### layers of model for validation data
        self.valid_layers, _, _ = new_cnn_fc_net(valid_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification')

        #### layers of model for test data
        self.test_layers, _, _ = new_cnn_fc_net(test_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification')

        #### functions of model
        tr_eval, v_eval, test_eval, tr_loss, v_loss, test_loss, tr_acc, v_acc, test_acc = mtl_model_output_functions(models=[[self.train_layers], [self.valid_layers], [self.test_layers]], y_batches=[[train_y_batch], [valid_y_batch], [test_y_batch]], num_tasks=1, dim_output=self.fc_size[-1], classification=True)

        self.train_eval, self.valid_eval, self.test_eval = tr_eval[0], v_eval[0], test_eval[0]
        self.train_loss, self.valid_loss, self.test_loss = tr_loss[0], v_loss[0], test_loss[0]
        self.train_accuracy, self.valid_accuracy, self.test_accuracy = tr_acc[0], v_acc[0], test_acc[0]

        #self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.loss)
        self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss)

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        cnn_param_val = get_value_of_valid_tensors(sess, self.cnn_param)
        fc_param_val = get_value_of_valid_tensors(sess, self.fc_param)

        parameters_val = {}
        parameters_val['conv_trainable_weights'] = savemat_wrapper(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper(fc_param_val)
        return parameters_val


###############################################################
#### Single task learner (CNN + FC) for Multi-task setting ####
###############################################################
#### Convolutional & Fully-connected Neural Net
class MTL_several_CNN_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernel, dim_strides, batch_size, learning_rate, learning_rate_decay, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, cnn_params=None, fc_params=None, saveData=False, skip_connect=[]):
        self.num_tasks = num_tasks
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel

        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        if cnn_params is None:
            cnn_params = [None for _ in range(self.num_tasks)]
        if fc_params is None:
            fc_params = [None for _ in range(self.num_tasks)]

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

        #### layers of model for train data
        self.train_models = []
        self.param, self.cnn_param, self.fc_param = [], [], []
        for task_cnt in range(self.num_tasks):
            model_tmp, cnn_param_tmp, fc_param_tmp = new_cnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size[task_cnt], cnn_params=cnn_params[task_cnt], fc_params=fc_params[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', skip_connections=list(skip_connect))
            self.train_models.append(model_tmp)
            self.cnn_param.append(cnn_param_tmp)
            self.fc_param.append(fc_param_tmp)
            self.param = self.param + cnn_param_tmp + fc_param_tmp

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _ = new_cnn_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size[task_cnt], cnn_params=self.cnn_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', skip_connections=list(skip_connect))
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _ = new_cnn_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size[task_cnt], cnn_params=self.cnn_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', skip_connections=list(skip_connect))
            self.test_models.append(model_tmp)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.fc_size[-1], classification=True)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]
        self.gradient = [tf.gradients(self.train_loss[x], self.cnn_param[x]+self.fc_param[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        cnn_param_val = [get_value_of_valid_tensors(sess, cnn_param) for cnn_param in self.cnn_param]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_param]

        parameters_val = {}
        parameters_val['conv_trainable_weights'] = savemat_wrapper_nested_list(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val

#### simplified NN model to compute saliency
class CNN_FC_for_saliency():
    def __init__(self, model_hyperpara, cnn_params=None, fc_params=None, detail=None):
        dim_channels = model_hyperpara['channel_sizes']
        dim_fcs = model_hyperpara['hidden_layer']
        input_size = model_hyperpara['image_dimension']
        dim_kernel = model_hyperpara['kernel_sizes']
        dim_strides = model_hyperpara['stride_sizes']
        padding_type = model_hyperpara['padding_type']
        max_pooling = model_hyperpara['max_pooling']
        dim_pool = model_hyperpara['pooling_size']
        dropout = model_hyperpara['dropout']
        skip_connect = model_hyperpara['skip_connect']

        assert (cnn_params is not None), "Must provide trained CNN parameters!!"
        assert (fc_params is not None), "Must provide trained FCN parameters!!"
        self.cnn_param, self.fc_param = cnn_params, fc_params

        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel

        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = dim_fcs
        self.input_size = input_size    ## [img_width, img_height, img_channel]
        self.detail = detail

        #### placeholder of model
        self.model_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size[0], self.input_size[1], self.input_size[2]])
        self.class_selector = tf.placeholder(tf.int32)

        #### layers of model
        self.model, _, _ = new_cnn_fc_net(self.model_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=cnn_params, fc_params=fc_params, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=1.0, input_size=self.input_size[0:2], output_type='classification', skip_connections=list(skip_connect), trainable=False)

        logits = self.model[-1]
        self.y = logits[0,self.class_selector]
        self.prob = tf.nn.softmax(logits)[0]
        self.pred = tf.argmax(logits, 1)[0]
        self.preds = tf.argmax(logits, 1)


########################################################
####     Single CNN + FC for Multi-task Learning    ####
########################################################
#### Convolutional & Fully-connected Neural Net
class MTL_CNN_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernel, dim_strides, batch_size, learning_rate, learning_rate_decay, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, saveData=False, skip_connect=[]):
        self.num_tasks = num_tasks
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
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

        #### layers of model for train data
        self.train_models = []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.cnn_param, self.fc_param = new_cnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', skip_connections=list(skip_connect))
            else:
                model_tmp, _, _ = new_cnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', skip_connections=list(skip_connect))
            self.train_models.append(model_tmp)
        self.param = self.cnn_param + self.fc_param

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _ = new_cnn_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', skip_connections=list(skip_connect))
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _ = new_cnn_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', skip_connections=list(skip_connect))
            self.test_models.append(model_tmp)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.fc_size[-1], classification=True)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        cnn_param_val = get_value_of_valid_tensors(sess, self.cnn_param)
        fc_param_val = get_value_of_valid_tensors(sess, self.fc_param)

        parameters_val = {}
        parameters_val['conv_trainable_weights'] = savemat_wrapper(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper(fc_param_val)
        return parameters_val


########################################################
####      Single CNN + Task-specific FC for MTL     ####
########         Hard Parameter Sharing         ########
########################################################
#### Convolutional & Fully-connected Neural Net
class MTL_CNN_HPS_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernel, dim_strides, batch_size, learning_rate, learning_rate_decay, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, saveData=False, skip_connect=[]):
        self.num_tasks = num_tasks
        #self.num_layers = [len(dim_fcs[0])] + [len(dim_fcs[1][x]) for x in range(self.num_tasks)]
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.num_TS_fc_param = 0
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
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

        #### layers of model for train data
        self.train_models, self.cnn_param, self.fc_param = new_hardparam_cnn_fc_nets(train_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.num_tasks, cnn_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', skip_connections=list(skip_connect))
        self.param = self.cnn_param + sum(self.fc_param, [])

        #### layers of model for validation data
        self.valid_models, _, _ = new_hardparam_cnn_fc_nets(valid_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.num_tasks, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', skip_connections=list(skip_connect))

        #### layers of model for test data
        self.test_models, _, _ = new_hardparam_cnn_fc_nets(test_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.num_tasks, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', skip_connections=list(skip_connect))

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.fc_size[0][-1], classification=True)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        cnn_param_val = get_value_of_valid_tensors(sess, self.cnn_param)
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_param]

        parameters_val = {}
        parameters_val['conv_trainable_weights'] = savemat_wrapper(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val


########################################################
####     CNN + FC model for Multi-task Learning     ####
########          Tensor Factorization          ########
########################################################
#### Convolutional & Fully-connected Neural Net
class MTL_CNN_tensorfactor_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernel, dim_strides, batch_size, learning_rate, learning_rate_decay, data_list, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, factor_type='Tucker', factor_eps_or_k=0.01, saveData=False, init_param=None, skip_connect=[]):
        self.num_tasks = num_tasks
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
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

        #### layers of model for train data
        self.train_models, self.cnn_param, self.fc_param = new_tensorfactored_cnn_fc_nets(train_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.num_tasks, cnn_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', factor_type=factor_type, factor_eps_or_k=factor_eps_or_k, init_param=init_param, skip_connections=list(skip_connect))
        self.param = self.cnn_param + self.fc_param

        #### layers of model for validation data
        self.valid_models, _, _ = new_tensorfactored_cnn_fc_nets(valid_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.num_tasks, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', factor_type=factor_type, factor_eps_or_k=factor_eps_or_k, skip_connections=list(skip_connect))

        #### layers of model for test data
        self.test_models, _, _ = new_tensorfactored_cnn_fc_nets(test_x_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.num_tasks, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', factor_type=factor_type, factor_eps_or_k=factor_eps_or_k, skip_connections=list(skip_connect))

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.fc_size[-1], classification=True)

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        cnn_param_val = get_value_of_valid_tensors(sess, self.cnn_param)
        fc_param_val = get_value_of_valid_tensors(sess, self.fc_param)

        parameters_val = {}
        parameters_val['conv_trainable_weights'] = savemat_wrapper(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper(fc_param_val)
        return parameters_val


########################################################
####     CNN + FC model for Multi-task Learning     ####
########         Progressive Neural Net         ########
########################################################
#### Convolutional & Fully-connected Neural Net
class MTL_CNN_progressive_minibatch():
    def __init__(self, dim_channels, dim_fcs, input_size, dim_kernel, dim_strides, batch_size, learning_rate, learning_rate_decay, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, dim_reduction_scale=1.0, skip_connect=[]):
        self.num_tasks = 1
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        self.padding_type = padding_type
        self.max_pooling = max_pooling
        self.dropout = dropout
        self.dim_reduction_scale=dim_reduction_scale
        self.skip_connect = skip_connect

        self.num_trainable_var = 0
        self.regenerate_network(prev_net_param=None)

    def regenerate_network(self, prev_net_param=None):
        # prev_net_param : [ [cnn_0, fc_0, lat_param_0], [cnn_1, fc_1, lat_param_1], ..., [net_param for num_tasks-2] ]
        assert ((prev_net_param is None and self.num_tasks == 1) or (prev_net_param is not None and self.num_tasks > 1)), "Parameters of previous columns are in wrong format"

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        self.model_input = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]])
        self.true_output = tf.placeholder(dtype=tf.float32, shape=[self.batch_size])
        self.model_input_batch = tf.reshape(self.model_input, [-1] + self.input_size)
        # model_input_batch instead of x_batch // true_output instead of y_batch

        #### layers of model for train data
        self.param, self.non_trainable_param = [], []
        self.train_models, self.cnn_params, self.cnn_lateral_params, self.fc_params = [], [], [], []
        # model with pre-trained and fixed param
        for task_cnt in range(self.num_tasks-1):
            # inside here, prev_net_param is not None
            if task_cnt < 1:
                model_tmp, cnn_param_tmp, cnn_lat_param_tmp, fc_param_tmp = new_progressive_cnn_fc_net(self.model_input_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=prev_net_param[task_cnt][0], fc_params=prev_net_param[task_cnt][1], padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', prev_net=None, cnn_lateral_params=None, trainable=False, dim_reduction_scale=self.dim_reduction_scale, skip_connections=list(self.skip_connect), use_numpy_var_in_graph=True)
            else:
                model_tmp, cnn_param_tmp, cnn_lat_param_tmp, fc_param_tmp = new_progressive_cnn_fc_net(self.model_input_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=prev_net_param[task_cnt][0], fc_params=prev_net_param[task_cnt][1], padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', prev_net=self.train_models, cnn_lateral_params=prev_net_param[task_cnt][2], trainable=False, dim_reduction_scale=self.dim_reduction_scale, skip_connections=list(self.skip_connect), use_numpy_var_in_graph=True)
            self.train_models.append(model_tmp)
            self.cnn_params.append(cnn_param_tmp)
            self.cnn_lateral_params.append(cnn_lat_param_tmp)
            self.fc_params.append(fc_param_tmp)
            self.non_trainable_param = self.non_trainable_param + [para if para is not None else [] for para in cnn_param_tmp] + [para if para is not None else [] for para in fc_param_tmp] + [para if para is not None else [] for para in cnn_lat_param_tmp]

        # model with trainable param
        if self.num_tasks < 2:
            model_tmp, cnn_param_tmp, cnn_lat_param_tmp, fc_param_tmp = new_progressive_cnn_fc_net(self.model_input_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=None, fc_params=None, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', prev_net=None, cnn_lateral_params=None, trainable=True, dim_reduction_scale=self.dim_reduction_scale, skip_connections=list(self.skip_connect))
        else:
            model_tmp, cnn_param_tmp, cnn_lat_param_tmp, fc_param_tmp = new_progressive_cnn_fc_net(self.model_input_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=None, fc_params=None, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', prev_net=self.train_models, cnn_lateral_params=None, trainable=True, dim_reduction_scale=self.dim_reduction_scale, skip_connections=list(self.skip_connect))
        self.train_models.append(model_tmp)
        self.cnn_params.append(cnn_param_tmp)
        self.cnn_lateral_params.append(cnn_lat_param_tmp)
        self.fc_params.append(fc_param_tmp)
        self.param = self.param + [para if para is not None else [] for para in cnn_param_tmp] + [para if para is not None else [] for para in fc_param_tmp] + [para if para is not None else [] for para in cnn_lat_param_tmp]

        #self.param = [self.cnn_params, self.fc_params, self.cnn_lateral_params]

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            if task_cnt < 1:
                model_tmp, _, _, _ = new_progressive_cnn_fc_net(self.model_input_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_params[task_cnt], fc_params=self.fc_params[task_cnt], padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', prev_net=None, cnn_lateral_params=self.cnn_lateral_params[task_cnt], trainable=False, dim_reduction_scale=self.dim_reduction_scale, skip_connections=list(self.skip_connect), use_numpy_var_in_graph=True)
            else:
                model_tmp, _, _, _ = new_progressive_cnn_fc_net(self.model_input_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_params[task_cnt], fc_params=self.fc_params[task_cnt], padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', prev_net=self.valid_models, cnn_lateral_params=self.cnn_lateral_params[task_cnt], trainable=False, dim_reduction_scale=self.dim_reduction_scale, skip_connections=list(self.skip_connect), use_numpy_var_in_graph=(task_cnt<self.num_tasks-1))
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            if task_cnt < 1:
                model_tmp, _, _, _ = new_progressive_cnn_fc_net(self.model_input_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_params[task_cnt], fc_params=self.fc_params[task_cnt], padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', prev_net=None, cnn_lateral_params=self.cnn_lateral_params[task_cnt], trainable=False, dim_reduction_scale=self.dim_reduction_scale, skip_connections=list(self.skip_connect), use_numpy_var_in_graph=True)
            else:
                model_tmp, _, _, _ = new_progressive_cnn_fc_net(self.model_input_batch, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_params[task_cnt], fc_params=self.fc_params[task_cnt], padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', prev_net=self.test_models, cnn_lateral_params=self.cnn_lateral_params[task_cnt], trainable=False, dim_reduction_scale=self.dim_reduction_scale, skip_connections=list(self.skip_connect), use_numpy_var_in_graph=(task_cnt<self.num_tasks-1))
            self.test_models.append(model_tmp)

        #### functions of model
        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy, self.train_output_label, self.valid_output_label, self.test_output_label = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [[self.true_output for _ in range(self.num_tasks)], [self.true_output for _ in range(self.num_tasks)], [self.true_output for _ in range(self.num_tasks)]], self.num_tasks, self.fc_size[-1], classification=True)

        #self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate / (1.0 + self.epoch * self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]
        self.update = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate / (1.0 + self.epoch * self.learn_rate_decay)).minimize(self.train_loss[-1])

        ### (Caution) No way to remove existing older computational graph, so TF stores all older networks
        self.num_trainable_var = self.num_trainable_var + count_trainable_var()
        #self.num_trainable_var = count_trainable_var()

    def get_prev_net_param(self, sess):
        # parameter of 0 ~ num_tasks-2 column : numpy array
        # parameter of num_tasks-1 column : tf variable
        prev_trained_net_param = []
        for task_cnt in range(self.num_tasks-1):
            prev_trained_net_param.append([self.cnn_params[task_cnt], self.fc_params[task_cnt], self.cnn_lateral_params[task_cnt]])

        prev_trained_net_param.append([[sess.run(x) for x in self.cnn_params[self.num_tasks-1]], [sess.run(x) for x in self.fc_params[self.num_tasks-1]], [(x if (x is None) else sess.run(x)) for x in self.cnn_lateral_params[self.num_tasks-1]]])
        return prev_trained_net_param

    def new_lifelong_task(self, sess=None, params=None):
        self.num_tasks = self.num_tasks + 1
        if self.num_tasks > 1 and params is None:
            # get pre-trained model param
            assert (sess is not None), "Unable to get numpy-ver parameter of the model, give correct tf session"
            # prev_net_param : [ [cnn_0, fc_0, lat_param_0], [cnn_1, fc_1, lat_param_1], ..., [net_param for num_tasks-2] ]
            prev_trained_net_param = []
            for task_cnt in range(self.num_tasks-2):
                # they are already list of numpy tensors
                prev_trained_net_param.append([self.cnn_params[task_cnt], self.fc_params[task_cnt], self.cnn_lateral_params[task_cnt]])

            prev_trained_net_param.append([[sess.run(x) for x in self.cnn_params[self.num_tasks-2]], [sess.run(x) for x in self.fc_params[self.num_tasks-2]], [(x if (x is None) else sess.run(x)) for x in self.cnn_lateral_params[self.num_tasks-2]]])
        elif params is not None:
            prev_trained_net_param = params
        else:
            prev_trained_net_param = None
        self.regenerate_network(prev_net_param=prev_trained_net_param)


#### Single-task version of DF-CNN.tc3 : no shared knowledgebase
class MTL_several_CNN_deconvtm_minibatch():
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
        self.train_models, self.cnn_KB_param, self.cnn_TS_param, self.cnn_gen_param, self.fc_param = [], [], [], [], []
        for task_cnt in range(self.num_tasks):
            model_tmp, cnn_KB_param_tmp, cnn_TS_param_tmp, cnn_gen_param_tmp, fc_param_tmp = new_ELLA_cnn_deconv_tensordot_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=None, cnn_TS_params=None, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))

            if task_cnt == 0:
                self.num_cnn_TS_para = len(cnn_TS_param_tmp)
                self.num_fc_para = len(fc_param_tmp)

            self.train_models.append(model_tmp)
            self.cnn_KB_param.append(cnn_KB_param_tmp)
            self.cnn_TS_param.append(cnn_TS_param_tmp)
            self.cnn_gen_param.append(cnn_gen_param_tmp)
            self.fc_param.append(fc_param_tmp)
        self.param = sum(self.cnn_KB_param, []) + sum(self.cnn_TS_param, []) + sum(self.fc_param, [])

        assert (all([(len(self.cnn_KB_param[x]) == self.num_cnn_KB_para) for x in range(len(self.cnn_TS_param))])), "CNN KB size doesn't match"
        assert (sum([len(self.cnn_TS_param[x]) for x in range(len(self.cnn_TS_param))]) == self.num_tasks * self.num_cnn_TS_para), "CNN TS size doesn't match"

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_tensordot_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param[task_cnt], cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_tensordot_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param[task_cnt], cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
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

        TS_grads = [tf.gradients(self.train_loss[x] + TS_reg_term1 + TS_reg_term2, self.cnn_TS_param[x]) for x in range(self.num_tasks)]
        TS_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(TS_grads[x], self.cnn_TS_param[x])] for x in range(self.num_tasks)]

        fc_grads = [tf.gradients(self.train_loss[x], self.fc_param[x]) for x in range(self.num_tasks)]
        fc_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(fc_grads[x], self.fc_param[x])] for x in range(self.num_tasks)]

        grads_vars = [KB_grads_vars[x] + TS_grads_vars[x] + fc_grads_vars[x] for x in range(self.num_tasks)]
        trainer = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate) for x in range(self.num_tasks)]
        self.update = [trainer[x].apply_gradients(grads_vars[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        KB_param_val = [get_value_of_valid_tensors(sess, cnn_kb_param) for cnn_kb_param in self.cnn_KB_param]
        TS_param_val = [get_value_of_valid_tensors(sess, cnn_ts_param) for cnn_ts_param in self.cnn_TS_param]
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.cnn_gen_param]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_param]

        parameters_val = {}
        parameters_val['conv_KB'] = savemat_wrapper_nested_list(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val



#### Single-task version of DF-CNN.tc2 : no shared knowledgebase
class MTL_several_CNN_deconvtm2_minibatch():
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
        self.train_models, self.cnn_KB_param, self.cnn_TS_param, self.cnn_gen_param, self.fc_param = [], [], [], [], []
        for task_cnt in range(self.num_tasks):
            model_tmp, cnn_KB_param_tmp, cnn_TS_param_tmp, cnn_gen_param_tmp, fc_param_tmp = new_ELLA_cnn_deconv_tensordot_fc_net2(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=None, cnn_TS_params=None, fc_params=None, KB_reg_type=KB_l2_reg, TS_reg_type=TS_l2_reg, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))

            if task_cnt == 0:
                self.num_cnn_TS_para = len(cnn_TS_param_tmp)
                self.num_fc_para = len(fc_param_tmp)

            self.train_models.append(model_tmp)
            self.cnn_KB_param.append(cnn_KB_param_tmp)
            self.cnn_TS_param.append(cnn_TS_param_tmp)
            self.cnn_gen_param.append(cnn_gen_param_tmp)
            self.fc_param.append(fc_param_tmp)
        self.param = sum(self.cnn_KB_param, []) + sum(self.cnn_TS_param, []) + sum(self.fc_param, [])

        assert (all([(len(self.cnn_KB_param[x]) == self.num_cnn_KB_para) for x in range(len(self.cnn_TS_param))])), "CNN KB size doesn't match"
        assert (sum([len(self.cnn_TS_param[x]) for x in range(len(self.cnn_TS_param))]) == self.num_tasks * self.num_cnn_TS_para), "CNN TS size doesn't match"

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_tensordot_fc_net2(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param[task_cnt], cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _, _, _ = new_ELLA_cnn_deconv_tensordot_fc_net2(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, self.cnn_KB_size, self.cnn_TS_size, self.cnn_deconv_stride_size, cnn_para_activation_fn=relation_activation_fn_cnn, cnn_KB_params=self.cnn_KB_param[task_cnt], cnn_TS_params=self.cnn_TS_param[task_cnt], fc_params=self.fc_param[task_cnt], padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', task_index=task_cnt, skip_connections=list(skip_connect))
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

        TS_grads = [tf.gradients(self.train_loss[x] + TS_reg_term1 + TS_reg_term2, self.cnn_TS_param[x]) for x in range(self.num_tasks)]
        TS_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(TS_grads[x], self.cnn_TS_param[x])] for x in range(self.num_tasks)]

        fc_grads = [tf.gradients(self.train_loss[x], self.fc_param[x]) for x in range(self.num_tasks)]
        fc_grads_vars = [[(grad/(1.0+self.epoch*self.learn_rate_decay), param) for grad, param in zip(fc_grads[x], self.fc_param[x])] for x in range(self.num_tasks)]

        grads_vars = [KB_grads_vars[x] + TS_grads_vars[x] + fc_grads_vars[x] for x in range(self.num_tasks)]
        trainer = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate) for x in range(self.num_tasks)]
        self.update = [trainer[x].apply_gradients(grads_vars[x]) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def get_params_val(self, sess):
        KB_param_val = [get_value_of_valid_tensors(sess, cnn_kb_param) for cnn_kb_param in self.cnn_KB_param]
        TS_param_val = [get_value_of_valid_tensors(sess, cnn_ts_param) for cnn_ts_param in self.cnn_TS_param]
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.cnn_gen_param]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_param]

        parameters_val = {}
        parameters_val['conv_KB'] = savemat_wrapper_nested_list(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val