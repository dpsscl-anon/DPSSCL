import tensorflow as tf
import numpy as np
import re, random, collections
from collections import defaultdict
from numpy import linalg as LA
from sklearn.metrics import roc_curve, auc

from os import getcwd, listdir, mkdir
import scipy.io as spio

#from tensorflow.contrib.memory_stats.python.ops import memory_stats_ops

from utils.utils import convert_array_to_oneHot



def save_param_to_mat(param_dict):
    param_to_save_format = {}
    for key, val in param_dict.items():
        scope_name = key.split(':')[0]
        scope = scope_name.split('/')[0]
        name = scope_name.split('/')[1]
        new_scope_name = scope + '_' + name
        param_to_save_format[new_scope_name] = val
    return param_to_save_format


def accuracy(preds, labels):
    return (100.0 * np.sum(np.argmax(preds, 1) == np.argmax(labels, 1)) / preds.shape[0])

def RMSE(p, y):
    N = p.shape[0]
    diff = p - y
    return np.sqrt((diff**2).mean())

def ROC_AUC(p, y):
    fpr, tpr, th = roc_curve(y, p)
    _auc = auc(fpr, tpr)
    _roc = (fpr, tpr)
    return _roc, _auc

class CNN_FC_DEN(object):
    def __init__(self, model_hyperpara, train_hyperpara, data_info):
        self.T = 0
        self.task_indices = []

        self.params = dict()
        self.param_trained = set()
        self.time_stamp = dict()

        self.batch_size = model_hyperpara['batch_size']
        self.input_shape = model_hyperpara['image_dimension']
        self.n_classes = data_info[2][0]
        self.ex_k = model_hyperpara['den_expansion']

        self.n_conv_layers = len(model_hyperpara['channel_sizes'])
        self.n_fc_layers = len(model_hyperpara['hidden_layer'])+1
        self.n_layers = self.n_conv_layers + self.n_fc_layers
        self.layer_types = ['input']+['conv' for _ in range(self.n_conv_layers)]+['fc' for _ in range(self.n_fc_layers)]

        self.cnn_kernel = model_hyperpara['kernel_sizes']
        self.cnn_stride = model_hyperpara['stride_sizes']
        self.cnn_channel = [self.input_shape[-1]]+model_hyperpara['channel_sizes']
        self.cnn_max_pool = model_hyperpara['max_pooling']
        self.cnn_pooling_sizes = model_hyperpara['pooling_size']
        self.cnn_dropout = model_hyperpara['dropout']
        self.cnn_padding_type = model_hyperpara['padding_type']
        self.cnn_output_dim = self.get_cnn_output_dims()
        self.cnn_output_spatial_dim = self.cnn_output_dim//self.cnn_channel[-1]
        self.fc_hiddens = [self.cnn_output_dim] + model_hyperpara['hidden_layer'] + [self.n_classes]

        self.l1_lambda = model_hyperpara['l1_lambda']
        self.l2_lambda = model_hyperpara['l2_lambda']
        self.gl_lambda = model_hyperpara['gl_lambda']
        self.regular_lambda = model_hyperpara['reg_lambda']
        self.loss_thr = model_hyperpara['loss_threshold']
        self.spl_thr = model_hyperpara['sparsity_threshold']
        self.scale_up = 15

        self.init_lr = train_hyperpara['lr']
        self.lr_decay = train_hyperpara['lr_decay']
        #self.max_iter = train_hyperpara['learning_step_max']
        #self.early_training = self.max_iter / 10.
        self.max_epoch_per_task = train_hyperpara['patience']
        self.num_training_epoch = 0
        self.train_iter_counter = 0
        self.task_change_epoch = [1]
        self.num_total_tasks = train_hyperpara['num_tasks']

        for i in range(self.n_layers-1):
            if self.layer_types[i+1] == 'conv':
                w = self.create_variable('layer%d'%(i+1), 'weight', self.cnn_kernel[2*i:2*(i+1)]+self.cnn_channel[i:i+2])
                b = self.create_variable('layer%d'%(i+1), 'biases', [self.cnn_channel[i+1]])
            elif self.layer_types[i+1] == 'fc':
                w = self.create_variable('layer%d'%(i+1), 'weight', [self.fc_hiddens[i-self.n_conv_layers], self.fc_hiddens[i+1-self.n_conv_layers]])
                b = self.create_variable('layer%d'%(i+1), 'biases', [self.fc_hiddens[i+1-self.n_conv_layers]])
            else:
                continue

        self.cur_W, self.prev_W = dict(), dict()

    def set_sess_config(self, config):
        self.sess_config = config

    def get_cnn_output_dims(self):
        h_tmp = tf.placeholder(tf.float32, [self.batch_size]+self.input_shape)
        for cnt in range(self.n_conv_layers):
            weight = tf.Variable(tf.truncated_normal(self.cnn_kernel[2*cnt:2*(cnt+1)]+self.cnn_channel[cnt:cnt+2], stddev=0.05))
            bias = tf.Variable(tf.constant(0.2, dtype=tf.float32, shape=[self.cnn_channel[cnt+1]]))
            h_tmp = tf.nn.conv2d(h_tmp, weight, strides=[1]+self.cnn_stride[2*cnt:2*(cnt+1)]+[1], padding=self.cnn_padding_type) + bias
            if self.cnn_max_pool:
                h_tmp = tf.nn.max_pool(h_tmp, ksize=[1]+self.cnn_pooling_sizes[2*cnt:2*(cnt+1)]+[1], strides=[1]+self.cnn_pooling_sizes[2*cnt:2*(cnt+1)]+[1], padding=self.cnn_padding_type)
        output_dim = int(h_tmp.shape[1]*h_tmp.shape[2]*h_tmp.shape[3])
        self.destroy_graph()
        return output_dim

    def get_params(self):
        """ Access the parameters """
        mdict = dict()
        for scope_name, param in self.params.items():
            w = self.sess.run(param)
            mdict[scope_name] = w
        return mdict

    def load_params(self, params, top = False, time = 999):
        """ parmas: it contains weight parameters used in network, like ckpt """
        self.params = dict()
        if top:
            # for last layer nodes
            for scope_name, param in params.items():
                scope_name = scope_name.split(':')[0]
                if ('layer%d'%self.n_layers in scope_name) and (('_%d'%self.T) in scope_name):
                    w = tf.get_variable(scope_name, initializer = param, trainable = True)
                    self.params[w.name] = w
                elif 'layer%d'%self.n_layers in scope_name:
                    w = tf.get_variable(scope_name, initializer = param, trainable = False)
                    self.params[w.name] = w
                else:
                    pass
            return

        if time == 1:
            self.prev_W = dict()
        for scope_name, param in params.items():
            trainable = True
            if time == 1 and 'layer%d'%self.n_layers not in scope_name:
                self.prev_W[scope_name] = param
            scope_name = scope_name.split(':')[0]
            scope = scope_name.split('/')[0]
            name = scope_name.split('/')[1]
            if (scope == 'layer%d'%self.n_layers) and ('_%d'%self.T) not in name:
                trainable = False
            if (scope in self.param_trained):
                trainable = False
            # current task is trainable
            w = tf.get_variable(scope_name, initializer = param, trainable = trainable)
            self.params[w.name] = w

    def num_trainable_var(self, params_list=None):
        if params_list is None:
            params_list = self.get_params()

        total_var_count = 0
        for _, var in params_list.items():
            total_var_count += np.prod(var.shape)
        return total_var_count

    def create_variable(self, scope, name, shape, trainable = True):
        with tf.variable_scope(scope):
            w = tf.get_variable(name, shape, trainable = trainable)
            if 'new' not in w.name:
                self.params[w.name] = w
        return w

    def get_variable(self, scope, name, trainable = True):
        with tf.variable_scope(scope, reuse = True):
            w = tf.get_variable(name, trainable = trainable)
            self.params[w.name] = w
        return w

    def extend_bottom(self, scope, ex_k = 10):
        """ bottom layer expansion. scope is range of layer """
        w = self.get_variable(scope, 'weight')
        b = self.get_variable(scope, 'biases')
        prev_dim = w.get_shape().as_list()[0:3]
        new_w = self.create_variable('new', 'bw', prev_dim+[ex_k])
        new_b = self.create_variable('new', 'bb', [ex_k])

        expanded_w = tf.concat([w, new_w], 3)
        expanded_b = tf.concat([b, new_b], 0)

        self.params[w.name] = expanded_w
        self.params[b.name] = expanded_b
        self.time_stamp['task%d_conv'%self.T][1] += ex_k
        return expanded_w, expanded_b

    def extend_top(self, scope, ex_k = 10):
        """ top layer expansion. scope is range of layer """
        if 'layer%d'%self.n_layers == scope:
            # extend for all task layer
            for i in self.task_indices:
                if i == self.T:
                    w = self.get_variable(scope, 'weight_%d'%i, True)
                    b = self.get_variable(scope, 'biases_%d'%i, True)
                    new_w = tf.get_variable('new/n%d'%i, [ex_k, self.n_classes], trainable = True)
                else:
                    w = self.get_variable(scope, 'weight_%d'%i, False)
                    b = self.get_variable(scope, 'biases_%d'%i, False)
                    new_w = tf.get_variable('new/n%d'%i, [ex_k, self.n_classes], initializer = tf.constant_initializer(0.0), trainable = False)

                expanded_w = tf.concat([w, new_w], 0)
                self.params[w.name] = expanded_w
                self.params[b.name] = b
            return expanded_w, b
        else:
            w = self.get_variable(scope, 'weight')
            b = self.get_variable(scope, 'biases')

            level = int(re.findall(r'layer(\d)', scope)[0])
            expanded_n_units = self.expansion_layer[self.n_layers-level-2] # top-down

            next_dim = w.get_shape().as_list()[1]
            new_w = tf.get_variable(scope + 'new_tw', [self.ex_k, next_dim], trainable = True)

            expanded_w = tf.concat([w, new_w], 0)
            self.params[w.name] = expanded_w
            self.params[b.name] = b
            return expanded_w, b

    def extend_param(self, scope, ex_k, l):
        if 'layer%d'%self.n_layers == scope:
            # output layer
            for i in self.task_indices:
                '''
                if i == self.T: # current task(fragile)
                    w = self.get_variable(scope, 'weight_%d'%i, True)
                    b = self.get_variable(scope, 'biases_%d'%i, True)
                    new_w = tf.get_variable('new_fc/n%d'%i, [ex_k, self.n_classes], trainable = True)
                else:
                    # previous tasks
                    w = self.get_variable(scope, 'weight_%d'%i, False)
                    b = self.get_variable(scope, 'biases_%d'%i, False)
                    new_w = tf.get_variable('new_fc/n%d'%i, [ex_k, self.n_classes], initializer = tf.constant_initializer(0.0), trainable = False)
                expanded_w = tf.concat([w, new_w], 0)
                '''
                if i == self.T: # current task(fragile)
                    w = self.get_variable(scope, 'weight_%d'%i, True)
                    b = self.get_variable(scope, 'biases_%d'%i, True)
                    new_w = tf.get_variable('new_fc/n%d'%i, [ex_k, self.n_classes], trainable = True)
                    expanded_w = tf.concat([w, new_w], 0)
                else:
                    # previous tasks
                    w = self.get_variable(scope, 'weight_%d'%i, False)
                    b = self.get_variable(scope, 'biases_%d'%i, False)
                    expanded_w = w
                self.params[w.name] = expanded_w
                self.params[b.name] = b
            return expanded_w, b
        else:
            # layers other than output
            w = self.get_variable(scope, 'weight')
            b = self.get_variable(scope, 'biases')

            if self.layer_types[l] == 'conv':
                prev_dim = w.get_shape().as_list()[:3]
                next_dim = w.get_shape().as_list()[3]
                # connect bottom to top
                new_w = self.create_variable(scope+'/new_conv', 'bw', prev_dim+[ex_k])
                new_b = self.create_variable(scope+'/new_conv', 'bb', [ex_k])

                expanded_w = tf.concat([w, new_w], 3)
                expanded_b = tf.concat([b, new_b], 0)
                # connect top to bottom
                new_w2 = self.create_variable(scope+'/new_conv', 'tw', prev_dim[0:2]+[ex_k, next_dim + ex_k])

                expanded_w = tf.concat([expanded_w, new_w2], 2)
                self.time_stamp['task%d_conv'%self.T][l] += ex_k
            elif self.layer_types[l] == 'fc':
                prev_dim = w.get_shape().as_list()[0]
                next_dim = w.get_shape().as_list()[1]
                # connect bottom to top
                new_w = self.create_variable(scope+'/new_fc', 'bw', [prev_dim, ex_k])
                new_b = self.create_variable(scope+'/new_fc', 'bb', [ex_k])

                expanded_w = tf.concat([w, new_w], 1)
                expanded_b = tf.concat([b, new_b], 0)
                # connect top to bottom
                if self.layer_types[l-1] == 'conv':
                    new_w2 = self.create_variable(scope+'/new_fc', 'tw', [self.cnn_output_spatial_dim*ex_k, next_dim + ex_k])
                    orig_ch_dim = prev_dim // self.cnn_output_spatial_dim
                    temp_expanded_w = tf.reshape(expanded_w, [self.cnn_output_spatial_dim, orig_ch_dim, next_dim+ex_k])
                    temp_new_w2 = tf.reshape(new_w2, [self.cnn_output_spatial_dim, ex_k, next_dim+ex_k])
                    expanded_w = tf.reshape(tf.concat([temp_expanded_w, temp_new_w2], axis=1), [-1, next_dim+ex_k])
                    self.time_stamp['task%d_fc'%self.T][l-self.n_conv_layers-1] += self.cnn_output_spatial_dim*ex_k
                else:
                    new_w2 = self.create_variable(scope+'/new_fc', 'tw', [ex_k, next_dim + ex_k])
                    expanded_w = tf.concat([expanded_w, new_w2], 0)
                self.time_stamp['task%d_fc'%self.T][l-self.n_conv_layers] += ex_k

                #expanded_w = tf.concat([expanded_w, new_w2], 0)
            self.params[w.name] = expanded_w
            self.params[b.name] = expanded_b
            return expanded_w, expanded_b

    def build_model(self, task_id, prediction = False, splitting = False, expansion = None):
        bottom = tf.reshape(self.X, [-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        if splitting:
            for i in range(1, self.n_layers):
                prev_w = np.copy(self.prev_W_split['layer%d'%i + '/weight:0'])
                cur_w = np.copy(self.cur_W['layer%d'%i + '/weight:0'])
                indices = self.unit_indices['layer%d'%i]
                next_dim = prev_w.shape[-1]
                is_fc_w = True if self.layer_types[i] == 'fc' else False
                if i >= 2 and i < self.n_layers:
                    below_dim = prev_w.shape[-2]
                    below_indices = self.unit_indices['layer%d'%(i-1)]

                    is_conv_fc = True if (self.layer_types[i-1] == 'conv' and is_fc_w) else False

                    if is_conv_fc:
                        lower_layer_shape = np.copy(self.prev_W_split['layer%d'%(i-1) + '/weight:0']).shape
                        below_dim = lower_layer_shape[-1]

                        prev_w_temp, cur_w_temp = [], []
                        for spatial_cnt in range(self.cnn_output_spatial_dim):
                            p_prev_ary, p_new_ary, c_prev_ary, c_new_ary = [], [], [], []
                            for j in range(below_dim):
                                if j in below_indices:
                                    p_prev_ary.append(prev_w[spatial_cnt*below_dim+j,:])
                                    p_new_ary.append(cur_w[spatial_cnt*below_dim+j,:])
                                    c_prev_ary.append(cur_w[spatial_cnt*below_dim+j,:])
                                    c_new_ary.append(cur_w[spatial_cnt*below_dim+j,:])
                                else:
                                    p_prev_ary.append(cur_w[spatial_cnt*below_dim+j,:])
                                    c_prev_ary.append(cur_w[spatial_cnt*below_dim+j,:])
                            prev_w_temp += p_prev_ary+p_new_ary
                            cur_w_temp += c_prev_ary+c_new_ary

                        prev_w = np.array(prev_w_temp).astype(np.float32)
                        cur_w = np.array(cur_w_temp).astype(np.float32)
                    else:
                        bottom_p_prev_ary, bottom_p_new_ary, bottom_c_prev_ary, bottom_c_new_ary = [], [], [], []
                        for j in range(below_dim):
                            if j in below_indices:
                                bottom_p_prev_ary.append(prev_w[j, :] if (is_fc_w and not is_conv_fc) else prev_w[:, :, j, :])
                                bottom_p_new_ary.append(cur_w[j, :] if (is_fc_w and not is_conv_fc) else cur_w[:, :, j, :])
                                bottom_c_prev_ary.append(cur_w[j, :] if (is_fc_w and not is_conv_fc) else cur_w[:, :, j, :])
                                bottom_c_new_ary.append(cur_w[j, :] if (is_fc_w and not is_conv_fc) else cur_w[:, :, j, :])
                            else:
                                bottom_p_prev_ary.append(cur_w[j, :] if (is_fc_w and not is_conv_fc) else cur_w[:, :, j, :])
                                bottom_c_prev_ary.append(cur_w[j, :] if (is_fc_w and not is_conv_fc) else cur_w[:, :, j, :])
                        prev_w = np.array( bottom_p_prev_ary + bottom_p_new_ary ).astype(np.float32) if (is_fc_w and not is_conv_fc) else np.array( np.stack(bottom_p_prev_ary+bottom_p_new_ary, axis=2) ).astype(np.float32)
                        cur_w = np.array( bottom_c_prev_ary + bottom_c_new_ary ).astype(np.float32) if (is_fc_w and not is_conv_fc) else np.array( np.stack(bottom_c_prev_ary+bottom_c_new_ary, axis=2) ).astype(np.float32)

                prev_ary = []
                new_ary = []
                for j in range(next_dim):
                    if j in indices:
                        prev_ary.append(prev_w[:, j] if is_fc_w else prev_w[:, :, :, j])
                        new_ary.append(cur_w[:, j] if is_fc_w else cur_w[:, :, :, j]) # will be expanded
                    else:
                        prev_ary.append(cur_w[:, j] if is_fc_w else cur_w[:, :, :, j])
                # fully connected, L1
                expanded_w = np.array( prev_ary + new_ary ).T.astype(np.float32) if is_fc_w else np.array( np.stack(prev_ary+new_ary, axis=3) ).astype(np.float32)
                expanded_b = np.concatenate((self.prev_W_split['layer%d'%i + '/biases:0'], np.random.rand(len(new_ary)))).astype(np.float32)
                with tf.variable_scope('layer%d'%i):
                    w = tf.get_variable('weight', initializer = expanded_w, trainable = True)
                    b = tf.get_variable('biases', initializer = expanded_b, trainable = True)
                self.params[w.name] = w
                self.params[b.name] = b

                if self.layer_types[i] == 'conv':
                    bottom = tf.nn.relu(tf.nn.conv2d(bottom, w, strides=[1]+self.cnn_stride[2*(i-1):2*i]+[1], padding=self.cnn_padding_type) + b)
                    if self.cnn_max_pool:
                        bottom = tf.nn.max_pool(bottom, ksize=[1]+self.cnn_pooling_sizes[2*(i-1):2*i]+[1], strides=[1]+self.cnn_pooling_sizes[2*(i-1):2*i]+[1], padding=self.cnn_padding_type)
                    if i == self.n_conv_layers:
                        bottom = tf.contrib.layers.flatten(bottom)
                else:
                    bottom = tf.nn.relu(tf.matmul(bottom, w) + b)
            w, b = self.extend_top('layer%d'%self.n_layers, len(new_ary))
            self.y = tf.matmul(bottom, w) + b
        elif expansion:
            for i in range(1, self.n_layers):
                if i == 1:
                    w, b = self.extend_bottom('layer%d'%i, self.ex_k)
                else:
                    w, b = self.extend_param('layer%d'%i, self.ex_k, i)

                if self.layer_types[i] == 'conv':
                    bottom = tf.nn.relu(tf.nn.conv2d(bottom, w, strides=[1]+self.cnn_stride[2*(i-1):2*i]+[1], padding=self.cnn_padding_type) + b)
                    if self.cnn_max_pool:
                        bottom = tf.nn.max_pool(bottom, ksize=[1]+self.cnn_pooling_sizes[2*(i-1):2*i]+[1], strides=[1]+self.cnn_pooling_sizes[2*(i-1):2*i]+[1], padding=self.cnn_padding_type)
                    if i == self.n_conv_layers:
                        bottom = tf.contrib.layers.flatten(bottom)
                else:
                    bottom = tf.nn.relu(tf.matmul(bottom, w) + b)
            w, b = self.extend_param('layer%d'%self.n_layers, self.ex_k, self.n_layers)
            self.y = tf.matmul(bottom, w) + b
        elif prediction:
            stamp_conv = self.time_stamp['task%d_conv'%task_id]
            stamp_fc = self.time_stamp['task%d_fc'%task_id]
            for i in range(1, self.n_layers):
                w = self.get_variable('layer%d'%i, 'weight', False)
                b = self.get_variable('layer%d'%i, 'biases', False)
                if self.layer_types[i] == 'conv':
                    w = w[:,:, :stamp_conv[i-1], :stamp_conv[i]]
                    b = b[:stamp_conv[i]]
                elif self.layer_types[i-1] == 'conv':
                    hidden_dim = w.get_shape().as_list()[-1]
                    temp_w = tf.reshape(w, [self.cnn_output_spatial_dim, -1, hidden_dim])
                    w = tf.reshape(temp_w[:, :stamp_conv[i-1], :stamp_fc[i-self.n_conv_layers]], [-1, hidden_dim])
                    b = b[:stamp_fc[i-self.n_conv_layers]]
                else:
                    w = w[:stamp_fc[i-1-self.n_conv_layers], :stamp_fc[i-self.n_conv_layers]]
                    b = b[:stamp_fc[i-self.n_conv_layers]]
                print(' [*] layer %d, shape : %s'%(i, w.get_shape().as_list()))

                if self.layer_types[i] == 'conv':
                    bottom = tf.nn.relu(tf.nn.conv2d(bottom, w, strides=[1]+self.cnn_stride[2*(i-1):2*i]+[1], padding=self.cnn_padding_type) + b)
                    if self.cnn_max_pool:
                        bottom = tf.nn.max_pool(bottom, ksize=[1]+self.cnn_pooling_sizes[2*(i-1):2*i]+[1], strides=[1]+self.cnn_pooling_sizes[2*(i-1):2*i]+[1], padding=self.cnn_padding_type)
                    if i == self.n_conv_layers:
                        bottom = tf.contrib.layers.flatten(bottom)
                else:
                    bottom = tf.nn.relu(tf.matmul(bottom, w) + b)

            w = self.get_variable('layer%d'%self.n_layers, 'weight_%d'%task_id, False)
            b = self.get_variable('layer%d'%self.n_layers, 'biases_%d'%task_id, False)
            w = w[:stamp_fc[self.n_fc_layers-1], :stamp_fc[self.n_fc_layers]]
            b = b[:stamp_fc[self.n_fc_layers]]
            self.y = tf.matmul(bottom, w) + b
        else:
            for i in range(1, self.n_layers):
                w = self.get_variable('layer%d'%i, 'weight', True)
                b = self.get_variable('layer%d'%i, 'biases', True)

                if self.layer_types[i] == 'conv':
                    bottom = tf.nn.relu(tf.nn.conv2d(bottom, w, strides=[1]+self.cnn_stride[2*(i-1):2*i]+[1], padding=self.cnn_padding_type) + b)
                    if self.cnn_max_pool:
                        bottom = tf.nn.max_pool(bottom, ksize=[1]+self.cnn_pooling_sizes[2*(i-1):2*i]+[1], strides=[1]+self.cnn_pooling_sizes[2*(i-1):2*i]+[1], padding=self.cnn_padding_type)
                    if i == self.n_conv_layers:
                        bottom = tf.contrib.layers.flatten(bottom)
                else:
                    bottom = tf.nn.relu(tf.matmul(bottom, w) + b)
            prev_dim = bottom.get_shape().as_list()[1]
            w = self.create_variable('layer%d'%self.n_layers, 'weight_%d'%task_id, [prev_dim, self.n_classes], True)
            b = self.create_variable('layer%d'%self.n_layers, 'biases_%d'%task_id, [self.n_classes], True)
            self.y = tf.matmul(bottom, w) + b

        self.yhat = tf.nn.softmax(self.y)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.y, labels = tf.cast(self.Y, tf.int32)))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.cast(self.Y, tf.int64)), tf.float32))

        ## for prediction/accuracy during training
        if not prediction:
            reshaped_X = tf.reshape(self.X, [-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
            self.taskwise_accuracy = []
            for task_cnt in range(1, task_id+1):
                bottom = reshaped_X
                stamp_conv = self.time_stamp['task%d_conv'%task_cnt]
                stamp_fc = self.time_stamp['task%d_fc'%task_cnt]
                for i in range(1, self.n_layers):
                    w = self.params['layer%d/weight:0'%i]
                    b = self.params['layer%d/biases:0'%i]
                    if self.layer_types[i] == 'conv':
                        w = w[:,:, :stamp_conv[i-1], :stamp_conv[i]]
                        b = b[:stamp_conv[i]]
                    elif self.layer_types[i-1] == 'conv':
                        hidden_dim = w.get_shape().as_list()[-1]
                        temp_w = tf.reshape(w, [self.cnn_output_spatial_dim, -1, hidden_dim])
                        w = tf.reshape(temp_w[:, :stamp_conv[i-1], :stamp_fc[i-self.n_conv_layers]], [-1, stamp_fc[i-self.n_conv_layers]])
                        b = b[:stamp_fc[i-self.n_conv_layers]]
                    else:
                        w = w[:stamp_fc[i-1-self.n_conv_layers], :stamp_fc[i-self.n_conv_layers]]
                        b = b[:stamp_fc[i-self.n_conv_layers]]

                    if self.layer_types[i] == 'conv':
                        bottom = tf.nn.relu(tf.nn.conv2d(bottom, w, strides=[1]+self.cnn_stride[2*(i-1):2*i]+[1], padding=self.cnn_padding_type) + b)
                        if self.cnn_max_pool:
                            bottom = tf.nn.max_pool(bottom, ksize=[1]+self.cnn_pooling_sizes[2*(i-1):2*i]+[1], strides=[1]+self.cnn_pooling_sizes[2*(i-1):2*i]+[1], padding=self.cnn_padding_type)
                        if i == self.n_conv_layers:
                            bottom = tf.contrib.layers.flatten(bottom)
                    else:
                        bottom = tf.nn.relu(tf.matmul(bottom, w) + b)

                w = self.params['layer%d/weight_%d:0'%(self.n_layers, task_cnt)]
                b = self.params['layer%d/biases_%d:0'%(self.n_layers, task_cnt)]
                w = w[:stamp_fc[self.n_fc_layers-1], :stamp_fc[self.n_fc_layers]]
                b = b[:stamp_fc[self.n_fc_layers]]
                temp_y = tf.matmul(bottom, w) + b
                self.taskwise_accuracy.append(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(temp_y, 1), tf.cast(self.Y, tf.int64)), tf.float32)))

    def selective_learning(self, task_id, selected_params):
        bottom = tf.reshape(self.X, [-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        for i in range(1, self.n_layers):
            with tf.variable_scope('layer%d'%i):
                w = tf.get_variable('weight', initializer = selected_params['layer%d/weight:0'%i])
                b = tf.get_variable('biases', initializer = selected_params['layer%d/biases:0'%i])

            if self.layer_types[i] == 'conv':
                bottom = tf.nn.relu(tf.nn.conv2d(bottom, w, strides=[1]+self.cnn_stride[2*(i-1):2*i]+[1], padding=self.cnn_padding_type) + b)
                if self.cnn_max_pool:
                    bottom = tf.nn.max_pool(bottom, ksize=[1]+self.cnn_pooling_sizes[2*(i-1):2*i]+[1], strides=[1]+self.cnn_pooling_sizes[2*(i-1):2*i]+[1], padding=self.cnn_padding_type)
                if i == self.n_conv_layers:
                    bottom = tf.contrib.layers.flatten(bottom)
            else:
                bottom = tf.nn.relu(tf.matmul(bottom, w) + b)
        #last layer
        with tf.variable_scope('layer%d'%self.n_layers):
            for task_cnt in range(1, task_id):
                _ = tf.get_variable('weight_%d'%task_cnt, initializer = selected_params['layer%d/weight_%d:0'%(self.n_layers, task_cnt)], trainable=False)
                _ = tf.get_variable('biases_%d'%task_cnt, initializer = selected_params['layer%d/biases_%d:0'%(self.n_layers, task_cnt)], trainable=False)
            w = tf.get_variable('weight_%d'%task_id, initializer = selected_params['layer%d/weight_%d:0'%(self.n_layers, task_id)])
            b = tf.get_variable('biases_%d'%task_id, initializer = selected_params['layer%d/biases_%d:0'%(self.n_layers, task_id)])

        self.y = tf.matmul(bottom, w) + b
        self.yhat = tf.nn.softmax(self.y)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.y, labels = tf.cast(self.Y, tf.int32)))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.cast(self.Y, tf.int64)), tf.float32))

        ## for prediction/accuracy during training
        reshaped_X = tf.reshape(self.X, [-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        self.taskwise_accuracy = []

        for task_cnt in range(1, task_id+1):
            bottom = reshaped_X
            stamp_conv = self.time_stamp['task%d_conv'%task_cnt]
            stamp_fc = self.time_stamp['task%d_fc'%task_cnt]
            for i in range(1, self.n_layers):
                w = self.get_variable('layer%d'%i, 'weight', False)
                b = self.get_variable('layer%d'%i, 'biases', False)

                if self.layer_types[i] == 'conv':
                    c1, c2 = w.get_shape().as_list()[2:]
                    w = w[:,:, :min(stamp_conv[i-1], c1), :min(stamp_conv[i], c2)]
                    b = b[:min(stamp_conv[i], c2)]
                elif self.layer_types[i-1] == 'conv':
                    h2 = w.get_shape().as_list()[-1]
                    temp_w = tf.reshape(w, [self.cnn_output_spatial_dim, -1, h2])
                    temp_c = temp_w.get_shape().as_list()[1]
                    w = tf.reshape(temp_w[:, :min(stamp_conv[i-1], temp_c), :min(stamp_fc[i-self.n_conv_layers], h2)], [-1, min(stamp_fc[i-self.n_conv_layers], h2)])
                    b = b[:min(stamp_fc[i-self.n_conv_layers], h2)]
                else:
                    h1, h2 = w.get_shape().as_list()
                    w = w[:min(stamp_fc[i-1-self.n_conv_layers], h1), :min(stamp_fc[i-self.n_conv_layers], h2)]
                    b = b[:min(stamp_fc[i-self.n_conv_layers], h2)]

                if self.layer_types[i] == 'conv':
                    bottom = tf.nn.relu(tf.nn.conv2d(bottom, w, strides=[1]+self.cnn_stride[2*(i-1):2*i]+[1], padding=self.cnn_padding_type) + b)
                    if self.cnn_max_pool:
                        bottom = tf.nn.max_pool(bottom, ksize=[1]+self.cnn_pooling_sizes[2*(i-1):2*i]+[1], strides=[1]+self.cnn_pooling_sizes[2*(i-1):2*i]+[1], padding=self.cnn_padding_type)
                    if i == self.n_conv_layers:
                        bottom = tf.contrib.layers.flatten(bottom)
                else:
                    bottom = tf.nn.relu(tf.matmul(bottom, w) + b)

            w = self.get_variable('layer%d'%self.n_layers, 'weight_%d'%task_cnt, False)
            b = self.get_variable('layer%d'%self.n_layers, 'biases_%d'%task_cnt, False)
            w = w[:min(stamp_fc[self.n_fc_layers-1], h2), :stamp_fc[self.n_fc_layers]]
            b = b[:stamp_fc[self.n_fc_layers]]
            temp_y = tf.matmul(bottom, w) + b
            self.taskwise_accuracy.append(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(temp_y, 1), tf.cast(self.Y, tf.int64)), tf.float32)))

    def optimization(self, prev_W, selective = False, splitting = False, expansion = None):
        if selective:
            all_var = [ var for var in tf.trainable_variables() if 'layer%d'%self.n_layers in var.name ]
        else:
            all_var = [ var for var in tf.trainable_variables() if 'weight' in var.name or 'new_fc' in var.name or 'new_conv' in var.name]

        l2_losses = []
        for var in all_var:
            l2_losses.append(tf.nn.l2_loss(var))

        regular_terms = []
        if (not splitting) and (expansion is None):
            for var in all_var:
                if var.name in prev_W.keys():
                    prev_w = prev_W[var.name]
                    regular_terms.append(tf.nn.l2_loss(var-prev_w))
        else:
            for var in all_var:
                if var.name in prev_W.keys():
                    prev_w = prev_W[var.name]
                    if len(prev_w.shape) == 1:
                        sliced = var[:prev_w.shape[0]]
                    elif len(prev_w.shape) == 2:
                        sliced = var[:prev_w.shape[0], :prev_w.shape[1]]
                    else:
                        sliced = var[:prev_w.shape[0], :prev_w.shape[1], :prev_w.shape[2], :prev_w.shape[3]]
                    regular_terms.append(tf.nn.l2_loss( sliced - prev_w ))

        losses = self.loss + self.l2_lambda * tf.reduce_sum(l2_losses) + self.regular_lambda * tf.reduce_sum(regular_terms)

        opt = tf.train.AdamOptimizer(self.lr).minimize(losses, global_step = self.g_step)

        l1_var = [ var for var in tf.trainable_variables() if 'layer' in var.name or 'new_fc' in var.name or 'new_conv' in var.name ]

        l1_op_list = []
        l1_hyp = self.l1_lambda
        with tf.control_dependencies([opt]):
            for var in l1_var:
                th_t = tf.fill(tf.shape(var), tf.convert_to_tensor(l1_hyp))
                zero_t = tf.zeros(tf.shape(var))
                var_temp = var - (th_t * tf.sign(var))
                l1_op = var.assign(tf.where(tf.less(tf.abs(var), th_t), zero_t, var_temp))
                l1_op_list.append(l1_op)

        GL_conv_var = [ var for var in tf.trainable_variables() if 'new_conv/tw' in var.name or 'new_conv/bw' in var.name ]
        GL_fc_var = [ var for var in tf.trainable_variables() if 'new_fc/tw' in var.name or 'new_fc/bw' in var.name ]

        gl_op_list = []

        with tf.control_dependencies([opt]):
            # group lasso for conv layer
            for var in GL_conv_var:
                g_sum = tf.sqrt(tf.reduce_sum(tf.square(var), [k for k in range(len(var.get_shape())-1)]))
                th_t = self.lr * self.gl_lambda
                gw = []
                for i in range(var.get_shape()[3]):
                    temp_gw = var[:, :, :, i] - (th_t * var[:, :, :, i] / g_sum[i])
                    gw_gl = tf.where(tf.less(g_sum[i], th_t), tf.zeros(tf.shape(var[:, :, :, i])), temp_gw)
                    gw.append(gw_gl)
                gl_op = var.assign(tf.stack(gw, 3))
                gl_op_list.append(gl_op)

        with tf.control_dependencies([opt]):
            # group lasso for fc layer
            for var in GL_fc_var:
                g_sum = tf.sqrt(tf.reduce_sum(tf.square(var), 0))
                th_t = self.lr * self.gl_lambda
                gw = []
                for i in range(var.get_shape()[1]):
                    temp_gw = var[:, i] - (th_t * var[:, i] / g_sum[i])
                    gw_gl = tf.where(tf.less(g_sum[i], th_t), tf.zeros(tf.shape(var[:, i])), temp_gw)
                    gw.append(gw_gl)
                gl_op = var.assign(tf.stack(gw, 1))
                gl_op_list.append(gl_op)

        with tf.control_dependencies(l1_op_list + gl_op_list):
            self.opt = tf.no_op()


    def set_initial_states(self, decay_step, g_step_init=0.0):
        self.g_step = tf.Variable(g_step_init, trainable=False)
        '''
        self.lr = tf.train.exponential_decay(
                    self.init_lr,           # Base learning rate.
                    self.g_step * self.batch_size,  # Current index into the dataset.
                    decay_step,          # Decay step.
                    0.999,                # Decay rate.
                    staircase=True)
        '''
        self.lr = self.init_lr/(1.0+self.num_training_epoch*self.lr_decay)
        #self.X = tf.placeholder(tf.float32, [None, self.input_shape[0]*self.input_shape[1]*self.input_shape[2]])
        self.X = tf.placeholder(tf.float32, [self.batch_size, self.input_shape[0]*self.input_shape[1]*self.input_shape[2]])
        #self.Y = tf.placeholder(tf.float32, [None, self.n_classes])
        #self.Y = tf.placeholder(tf.float32, [None])
        self.Y = tf.placeholder(tf.float32, [self.batch_size])

    def add_task(self, task_id, data, saveParam=False, saveGraph=False):
        def get_unexceed_useless_indices(orig_weight_size, orig_useless_indices):
            return_list = []
            for a in orig_useless_indices:
                if a < orig_weight_size:
                    return_list.append(a)
            return return_list

        trainX, trainY, valX, valY, testX, testY = data
        data_size = len(trainY[self.T-1])
        self.train_range = np.arange(data_size)
        self.num_batches = data_size//self.batch_size
        self.train_iter_counter = 0
        self.max_iter = self.num_batches*self.max_epoch_per_task
        self.early_training = self.max_iter / 10.
        self.set_initial_states(data_size, g_step_init=float(self.num_batches*self.num_training_epoch))

        expansion_layer = [] # to split
        self.expansion_layer = [0 for _ in range(self.n_layers)] # new units
        self.best_valid_acc, self.test_acc_at_best_epoch = 0.0, 0.0

        ## initialization of time_stamp
        if task_id == 1:
            self.time_stamp['task%d_conv'%task_id] = list(self.cnn_channel)
            self.time_stamp['task%d_fc'%task_id] = list(self.fc_hiddens)
        else:
            self.time_stamp['task%d_conv'%task_id] = list(self.time_stamp['task%d_conv'%(task_id-1)])
            self.time_stamp['task%d_fc'%task_id] = list(self.time_stamp['task%d_fc'%(task_id-1)])

        if saveParam:
            param_save_path = getcwd() + '/DEN_params'
            if ('DEN_params' not in listdir(getcwd())):
                mkdir(getcwd()+'/DEN_params')

        if self.T == 1:
            self.build_model(task_id)
            self.optimization(self.prev_W)
            self.sess.run(tf.global_variables_initializer())
            #print("\n\nMemory Usages!!")
            #print("\tMem Limit:")
            #print(self.sess.run(memory_stats_ops.BytesLimit()))
            #print("\tMem In Use:")
            #print(self.sess.run(memory_stats_ops.BytesInUse()))
            #print("\tMem Max In Use:")
            #print(self.sess.run(memory_stats_ops.MaxBytesInUse()))
            #print("\n\n")

            if saveParam:
                params = self.get_params()
                spio.savemat(param_save_path+'/t%d_1_init.mat'%(self.T), {'parameter': save_param_to_mat(params)})

            train_acc, valid_acc, test_acc = self.compute_accuracy_during_training(data)
            train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy = [train_acc], [valid_acc], [test_acc], [0.0]

            repeated, _, hist_acc_temp = self.run_epoch(self.opt, self.loss, data, 'Train')
            train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy = self.reformat_accuracy_history((train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy), hist_acc_temp)
            expansion_layer = [0 for _ in range(self.n_layers)]

            if saveParam:
                params = self.get_params()
                spio.savemat(param_save_path+'/t%d_2_afterTraining.mat'%(self.T), {'parameter': save_param_to_mat(params)})

            if saveGraph:
                tfboard_writer = tf.summary.FileWriter('./graphs/DEN/task_'+str(self.T), self.sess.graph)
                tfboard_writer.close()

        else:
            """ SELECTIVE LEARN """
            print(' [*] Selective retraining')
            self.build_model(task_id)
            self.optimization(self.prev_W, selective = True)
            self.sess.run(tf.global_variables_initializer())

            train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy = [], [], [], []

            repeated, c_loss, hist_acc_temp = self.run_epoch(self.opt, self.loss, data, 'Train', selective = True, s_iter = 0)
            train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy = self.reformat_accuracy_history((train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy), hist_acc_temp)

            params = self.get_params()
            if saveParam:
                spio.savemat(param_save_path+'/t%d_1_lastWeight.mat'%(self.T), {'parameter': save_param_to_mat(params)})

            if saveGraph:
                tfboard_writer = tf.summary.FileWriter('./graphs/DEN/task_'+str(self.T)+'_1LastLayer', self.sess.graph)
                tfboard_writer.close()

            self.destroy_graph()
            self.sess.close()
            self.sess = tf.Session(config=self.sess_config)

            # select the units
            selected_prev_params = dict()
            selected_params = dict()
            all_indices = defaultdict(list) # nonzero unis
            for i in range(self.n_layers, 0, -1):
                if i == self.n_layers:
                    w = params['layer%d/weight_%d:0'%(i, task_id)]
                    b = params['layer%d/biases_%d:0'%(i, task_id)]
                    for j in range(w.shape[0]):
                        if w[j, 0] != 0:
                            all_indices['fclayer%d'%i].append(j)
                    selected_params['layer%d/weight_%d:0'%(i, task_id)] = w[np.ix_(all_indices['fclayer%d'%i], [0])]
                    selected_params['layer%d/biases_%d:0'%(i, task_id)] = b
                elif self.layer_types[i] == 'fc' and not (self.layer_types[i-1] == 'conv'):
                    w = params['layer%d/weight:0'%i]
                    b = params['layer%d/biases:0'%i]
                    top_indices = all_indices['fclayer%d'%(i+1)]
                    for j in range(w.shape[0]):
                        if np.count_nonzero(w[j, top_indices]) != 0 or i == 1:
                            all_indices['fclayer%d'%i].append(j)

                    sub_weight = w[np.ix_(all_indices['fclayer%d'%i], top_indices)]
                    sub_biases = b[all_indices['fclayer%d'%(i+1)]]
                    selected_params['layer%d/weight:0'%i] = sub_weight
                    selected_params['layer%d/biases:0'%i] = sub_biases
                    selected_prev_params['layer%d/weight:0'%i] = self.prev_W['layer%d/weight:0'%i][np.ix_(all_indices['fclayer%d'%i], top_indices)]
                    selected_prev_params['layer%d/biases:0'%i] = self.prev_W['layer%d/biases:0'%i][all_indices['fclayer%d'%(i+1)]]
                elif self.layer_types[i] == 'fc':
                    ## boundary between conv and fc layer
                    w = params['layer%d/weight:0'%i]
                    b = params['layer%d/biases:0'%i]
                    top_indices = all_indices['fclayer%d'%(i+1)]

                    num_chs, hidden_dim = w.shape[0]//self.cnn_output_spatial_dim, w.shape[1]
                    temp_w = np.reshape(w, [self.cnn_output_spatial_dim, num_chs, hidden_dim])
                    temp_indices = []
                    for j in range(temp_w.shape[1]):
                        if np.count_nonzero(temp_w[:, j, top_indices]) != 0:
                            all_indices['convlayer%d'%(i)].append(j)
                            temp_indices += [j+num_chs*a for a in range(self.cnn_output_spatial_dim)]
                    temp_indices.sort()
                    all_indices['fclayer%d'%i] = list(temp_indices)

                    sub_weight = w[np.ix_(all_indices['fclayer%d'%i], top_indices)]
                    sub_biases = b[all_indices['fclayer%d'%(i+1)]]
                    selected_params['layer%d/weight:0'%i] = sub_weight
                    selected_params['layer%d/biases:0'%i] = sub_biases
                    selected_prev_params['layer%d/weight:0'%i] = self.prev_W['layer%d/weight:0'%i][np.ix_(all_indices['fclayer%d'%i], top_indices)]
                    selected_prev_params['layer%d/biases:0'%i] = self.prev_W['layer%d/biases:0'%i][all_indices['fclayer%d'%(i+1)]]
                else:
                    w = params['layer%d/weight:0'%i]
                    b = params['layer%d/biases:0'%i]
                    top_indices = all_indices['convlayer%d'%(i+1)]

                    for j in range(w.shape[2]):
                        if np.count_nonzero(w[:,:,j, top_indices]) != 0 or i == 1:
                            all_indices['convlayer%d'%i].append(j)

                    k_size1, k_size2 = w.shape[0], w.shape[1]
                    sub_weight = w[np.ix_(range(k_size1), range(k_size2), all_indices['convlayer%d'%i], top_indices)]
                    sub_biases = b[top_indices]
                    selected_params['layer%d/weight:0'%i] = sub_weight
                    selected_params['layer%d/biases:0'%i] = sub_biases
                    selected_prev_params['layer%d/weight:0'%i] = self.prev_W['layer%d/weight:0'%i][np.ix_(range(k_size1), range(k_size2), all_indices['convlayer%d'%i], top_indices)]
                    selected_prev_params['layer%d/biases:0'%i] = self.prev_W['layer%d/biases:0'%i][top_indices]

            for task_cnt in range(1, task_id):
                w = params['layer%d/weight_%d:0'%(self.n_layers, task_cnt)]
                b = params['layer%d/biases_%d:0'%(self.n_layers, task_cnt)]
                selected_params['layer%d/weight_%d:0'%(self.n_layers, task_cnt)] = w[:self.time_stamp['task%d_fc'%task_cnt][-2], :self.time_stamp['task%d_fc'%task_cnt][-1]]
                selected_params['layer%d/biases_%d:0'%(self.n_layers, task_cnt)] = b[:self.time_stamp['task%d_fc'%task_cnt][-1]]


            # learn only selected params
            self.set_initial_states(data_size, g_step_init=float(self.num_batches*self.num_training_epoch))
            self.selective_learning(task_id, selected_params)
            self.optimization(selected_prev_params)
            self.sess.run(tf.global_variables_initializer())

            repeated, c_loss, hist_acc_temp = self.run_epoch(self.opt, self.loss, data, 'Train', selective=True, s_iter=self.early_training, s_max=int(self.max_iter*0.4))
            train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy = self.reformat_accuracy_history((train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy), hist_acc_temp)
            _vars = [(var.name, self.sess.run(var)) for var in tf.trainable_variables() if 'layer' in var.name]

            if saveGraph:
                tfboard_writer = tf.summary.FileWriter('./graphs/DEN/task_'+str(self.T)+'_2SelectiveLearning', self.sess.graph)
                tfboard_writer.close()

            for item in _vars:
                key, values = item
                selected_params[key] = values

            # union
            for i in range(self.n_layers, 0, -1):
                if i == self.n_layers:
                    temp_weight = params['layer%d/weight_%d:0'%(i, task_id)]
                    temp_weight[np.ix_(all_indices['fclayer%d'%i], [0])] = selected_params['layer%d/weight_%d:0'%(i, task_id)]
                    params['layer%d/weight_%d:0'%(i, task_id)] = temp_weight
                    params['layer%d/biases_%d:0'%(i, task_id)] = selected_params['layer%d/biases_%d:0'%(i, task_id)]
                elif self.layer_types[i] == 'fc':
                    temp_weight = params['layer%d/weight:0'%i]
                    temp_biases = params['layer%d/biases:0'%i]
                    temp_weight[np.ix_(all_indices['fclayer%d'%i], all_indices['fclayer%d'%(i+1)])] = selected_params['layer%d/weight:0'%i]
                    temp_biases[all_indices['fclayer%d'%(i+1)]] = selected_params['layer%d/biases:0'%i]
                    params['layer%d/weight:0'%i] = temp_weight
                    params['layer%d/biases:0'%i] = temp_biases
                else:
                    temp_weight = params['layer%d/weight:0'%i]
                    temp_biases = params['layer%d/biases:0'%i]
                    k_size1, k_size2 = temp_weight.shape[0], temp_weight.shape[1]
                    temp_weight[np.ix_(range(k_size1), range(k_size2), all_indices['convlayer%d'%i], all_indices['convlayer%d'%(i+1)])] = selected_params['layer%d/weight:0'%i]
                    temp_biases[all_indices['convlayer%d'%(i+1)]] = selected_params['layer%d/biases:0'%i]
                    params['layer%d/weight:0'%i] = temp_weight
                    params['layer%d/biases:0'%i] = temp_biases

            if saveParam:
                spio.savemat(param_save_path+'/t%d_2_selectiveLearning.mat'%(self.T), {'parameter': save_param_to_mat(params)})


            """ Network Expansion """
            if c_loss < self.loss_thr:
                pass
            else:
                # addition
                self.destroy_graph()
                self.sess.close()
                self.sess = tf.Session(config=self.sess_config)
                self.load_params(params)
                self.set_initial_states(data_size, g_step_init=float(self.num_batches*self.num_training_epoch))
                self.build_model(task_id, expansion = True)
                self.optimization(self.prev_W, expansion = True)
                self.sess.run(tf.global_variables_initializer())

                if saveParam:
                    params = self.get_params()
                    spio.savemat(param_save_path+'/t%d_2.3_before_expansion.mat'%(self.T), {'parameter': save_param_to_mat(params)})

                print(' [*] Network expansion (training)')
                repeated, c_loss, hist_acc_temp = self.run_epoch(self.opt, self.loss, data, 'Train', s_max=int(self.max_iter*0.3))
                train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy = self.reformat_accuracy_history((train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy), hist_acc_temp)

                # delete useless params adding by addition.
                params = self.get_params()

                if saveParam:
                    spio.savemat(param_save_path+'/t%d_2.6_before_removing_useless.mat'%(self.T), {'parameter': save_param_to_mat(params)})

                if saveGraph:
                    tfboard_writer = tf.summary.FileWriter('./graphs/DEN/task_'+str(self.T)+'_3Expansion', self.sess.graph)
                    tfboard_writer.close()

                for i in range(self.n_layers-1, 0, -1):
                    prev_layer_weight = params['layer%d/weight:0'%i]
                    prev_layer_biases = params['layer%d/biases:0'%i]
                    useless = []
                    if self.layer_types[i] == 'fc':
                        for j in range(prev_layer_weight.shape[1] - self.ex_k, prev_layer_weight.shape[1]):
                            if np.count_nonzero(prev_layer_weight[:, j]) == 0:
                                useless.append(j)
                        cur_layer_weight = np.delete(prev_layer_weight, useless, axis = 1)
                    elif self.layer_types[i] == 'conv':
                        for j in range(prev_layer_weight.shape[3] - self.ex_k, prev_layer_weight.shape[3]):
                            if np.count_nonzero(prev_layer_weight[:, :, :, j]) == 0:
                                useless.append(j)
                        cur_layer_weight = np.delete(prev_layer_weight, useless, axis = 3)
                    cur_layer_biases = np.delete(prev_layer_biases, useless)
                    params['layer%d/weight:0'%i] = cur_layer_weight
                    params['layer%d/biases:0'%i] = cur_layer_biases

                    if self.layer_types[i+1] == 'fc':
                        remove_axis = 0
                    elif self.layer_types[i+1] == 'conv':
                        remove_axis = 2
                    if i == self.n_layers-1:
                        for t in self.task_indices:
                            prev_layer_weight = params['layer%d/weight_%d:0'%(i+1, t)]
                            #cur_layer_weight = np.delete(prev_layer_weight, useless, axis = remove_axis)
                            adjusted_useless = get_unexceed_useless_indices(prev_layer_weight.shape[remove_axis], useless)
                            cur_layer_weight = np.delete(prev_layer_weight, adjusted_useless, axis=remove_axis)
                            params['layer%d/weight_%d:0'%(i+1, t)] = cur_layer_weight
                    else:
                        prev_layer_weight = params['layer%d/weight:0'%(i+1)]
                        #cur_layer_weight = np.delete(prev_layer_weight, useless, axis = remove_axis)
                        adjusted_useless = get_unexceed_useless_indices(prev_layer_weight.shape[remove_axis], useless)
                        cur_layer_weight = np.delete(prev_layer_weight, adjusted_useless, axis=remove_axis)
                        params['layer%d/weight:0'%(i+1)] = cur_layer_weight

                    self.expansion_layer[i-1] = self.ex_k - len(useless)

                    print("   [*] Expanding %dth layer, %d unit added, (valid, repeated: %d)" %(i, self.expansion_layer[i-1], repeated))

                if saveParam:
                    spio.savemat(param_save_path+'/t%d_3_expansion.mat'%(self.T), {'parameter': save_param_to_mat(params)})

                if saveGraph:
                    tfboard_writer = tf.summary.FileWriter('./graphs/DEN/task_'+str(self.T)+'_4Pruning_after_Expansion', self.sess.graph)
                    tfboard_writer.close()


                print(' [*] Split & Duplication')
                self.cur_W = params
                # find the highly drifted ones and split
                self.unit_indices = dict()
                for i in range(1, self.n_layers):
                    prev = self.prev_W['layer%d/weight:0'%i]
                    cur = params['layer%d/weight:0'%i]
                    next_dim = prev.shape[-1]

                    indices = []
                    cosims = []

                    if self.layer_types[i] == 'fc':
                        spl_thr = self.spl_thr[1]
                        scale_up = self.scale_up
                    elif self.layer_types[i] == 'conv':
                        spl_thr = self.spl_thr[0]
                        scale_up = 1

                    for j in range(next_dim):
                        if self.layer_types[i] == 'fc':
                            prev_vec, cur_vec = prev[:, j], cur[:prev.shape[0], j]
                        elif self.layer_types[i] == 'conv':
                            prev_vec, cur_vec = prev[:,:,:,j].reshape(-1), cur[:prev.shape[0],:prev.shape[1],:prev.shape[2], j].reshape(-1)
                        cosim = np.abs(np.dot(prev_vec, cur_vec)+1e-7)/( LA.norm(prev_vec) * LA.norm(cur_vec) + 1e-7 )

                        if cosim < spl_thr:
                            indices.append(j)
                            cosims.append(cosim)
                    _temp = np.argsort(cosims)[:self.ex_k*scale_up]
                    print("   [*] split N in layer%d: %d / %d"%(i, len(_temp), len(cosims)))
                    indices = np.array(indices)[_temp]
                    self.expansion_layer[i-1] += len(indices)
                    expansion_layer.append(len(indices))
                    self.unit_indices['layer%d'%i] = indices

                self.prev_W_split = self.cur_W.copy()
                for key, values in self.prev_W.items():
                    temp = self.prev_W_split[key]
                    if len(values.shape) == 2:
                        temp[:values.shape[0], :values.shape[1]] = values
                    elif len(values.shape) == 4:
                        temp[:values.shape[0], :values.shape[1], :values.shape[2], :values.shape[3]] = values
                    else:
                        temp[:values.shape[0]] = values
                    self.prev_W_split[key] = temp

                if saveParam:
                    spio.savemat(param_save_path+'/t%d_4_split.mat'%(self.T), {'parameter': save_param_to_mat(params)})


                self.destroy_graph()
                self.sess.close()
                self.sess = tf.Session(config=self.sess_config)
                self.load_params(params, top = True)
                self.set_initial_states(data_size, g_step_init=float(self.num_batches*self.num_training_epoch))
                self.build_model(task_id, splitting = True)
                self.optimization(self.prev_W, splitting = True)
                self.sess.run(tf.global_variables_initializer())

                repeated, c_loss, hist_acc_temp = self.run_epoch(self.opt, self.loss, data, 'Train', s_max=int(self.max_iter*0.3))
                train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy = self.reformat_accuracy_history((train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy), hist_acc_temp)
                #print("   [*] split, loss: %.4f, nn_perf: %.4f(valid) repeated: %d"%(c_loss, val_perf, repeated))

        self.task_change_epoch.append(self.num_training_epoch+1)
        print("   [*] Total expansions: %s"%self.expansion_layer)

        params = self.get_params()
        stamp_conv, stamp_fc = [], []
        for i in range(1, self.n_layers+1):
            if self.layer_types[i] == 'conv':
                if i == 1:
                    stamp_conv.append(self.input_shape[-1])
                dim = params['layer%d/weight:0'%i].shape[3]
                stamp_conv.append(dim)
            else:
                if i == self.n_layers:
                    dim = params['layer%d/weight_%d:0'%(i, task_id)].shape[0]
                elif self.layer_types[i] == 'fc':
                    dim = params['layer%d/weight:0'%i].shape[0]
                stamp_fc.append(dim)

        stamp_fc.append(self.n_classes)
        self.time_stamp['task%d_conv'%task_id] = stamp_conv
        self.time_stamp['task%d_fc'%task_id] = stamp_fc

        for i in range(1, self.n_layers):
            self.param_trained.add('layer%d'%i)

        return train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy

    def run_epoch(self, opt, loss, data, desc = 'Train', selective = False, s_iter = 0, s_max=-1):
        trainX, trainY, valX, valY, testX, testY = data
        train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy = [], [], [], []
        c_iter = s_iter
        s_max = self.max_iter if s_max < 0 else s_max
        while(s_max > c_iter):
            batch_X, batch_Y = self.data_iteration(trainX[self.T-1], trainY[self.T-1], desc)
            _, c_loss = self.sess.run([opt, loss], feed_dict = {self.X: batch_X, self.Y: batch_Y})
            c_iter += 1
            self.train_iter_counter += 1
            if self.train_iter_counter%self.num_batches == 0:
                self.num_training_epoch += 1
                train_acc, valid_acc, test_acc = self.compute_accuracy_during_training(data)
                train_accuracy.append(train_acc)
                valid_accuracy.append(valid_acc)
                test_accuracy.append(test_acc)
                print('epoch %d - Train : %f, Validation : %f' % (self.num_training_epoch, train_acc[self.T-1], valid_acc[self.T-1]))
                if valid_acc[self.T-1] > self.best_valid_acc:
                    self.best_valid_acc = valid_acc[self.T-1]
                    self.test_acc_at_best_epoch = test_acc[self.T-1]
                    str_temp = '\t<<' if (valid_acc[self.T-1] > self.best_valid_acc * 1.002) else ''
                    print('\t\t\t\t\t\t\tTest : %f%s' % (abs(test_acc[self.T-1]), str_temp))
                best_test_accuracy.append(self.test_acc_at_best_epoch)


            if selective and s_iter < 1 and c_iter >= self.early_training:
                break

        return c_iter, c_loss, (train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy)

    def compute_accuracy_during_training(self, data):
        trainX, trainY, valX, valY, testX, testY = data
        train_acc, valid_acc, test_acc = np.zeros([self.num_total_tasks+1], dtype=np.float32), np.zeros([self.num_total_tasks+1], dtype=np.float32), np.zeros([self.num_total_tasks+1], dtype=np.float32)

        for task_cnt in range(1, self.T+1):
            num_train, num_valid, num_test = trainX[task_cnt-1].shape[0], valX[task_cnt-1].shape[0], testX[task_cnt-1].shape[0]
            tr_acc, v_acc, te_acc = [], [], []
            for batch_cnt in range(num_train//self.batch_size):
                tr_acc.append(self.sess.run(self.taskwise_accuracy[task_cnt-1], {self.X: trainX[task_cnt-1][batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size,:], self.Y: trainY[task_cnt-1][batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size]}))

            for batch_cnt in range(num_valid//self.batch_size):
                v_acc.append(self.sess.run(self.taskwise_accuracy[task_cnt-1], {self.X: valX[task_cnt-1][batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size,:], self.Y: valY[task_cnt-1][batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size]}))

            for batch_cnt in range(num_test//self.batch_size):
                te_acc.append(self.sess.run(self.taskwise_accuracy[task_cnt-1], {self.X: testX[task_cnt-1][batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size,:], self.Y: testY[task_cnt-1][batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size]}))
            train_acc[task_cnt-1], valid_acc[task_cnt-1], test_acc[task_cnt-1] = np.mean(np.array(tr_acc)), np.mean(np.array(v_acc)), np.mean(np.array(te_acc))
        train_acc[-1], valid_acc[-1], test_acc[-1] = np.mean(train_acc[:self.T]), np.mean(valid_acc[:self.T]), np.mean(test_acc[:self.T])
        return train_acc, valid_acc, test_acc

    def reformat_accuracy_history(self, acc_dest, acc_src):
        tr_acc_dest, v_acc_dest, te_acc_dest, best_te_acc_dest = acc_dest
        tr_acc_src, v_acc_src, te_acc_src, best_te_acc_src = acc_src
        return list(tr_acc_dest+tr_acc_src), list(v_acc_dest+v_acc_src), list(te_acc_dest+te_acc_src), list(best_te_acc_dest+best_te_acc_src)

    def data_iteration(self, X, Y, desc = 'Train'):
        if desc == 'Train':
            random.shuffle(self.train_range)
            b_idx = self.train_range[: self.batch_size]
            return X[b_idx], Y[b_idx]
        else:
            return X, Y

    def get_performance(self, p, y):
        y_onehot = convert_array_to_oneHot(y, self.n_classes)
        perf_list = []
        for _i in range(self.n_classes):
            roc, perf = ROC_AUC(p[:,_i], y_onehot[:,_i])
            perf_list.append(perf)

        return np.mean(perf_list)

    def predict_perform(self, task_id, X, Y, task_name = None):
        self.X = tf.placeholder(tf.float32, [None, self.input_shape[0]*self.input_shape[1]*self.input_shape[2]])
        #self.Y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.Y = tf.placeholder(tf.float32, [None])
        self.build_model(task_id, prediction = True)
        self.sess.run(tf.global_variables_initializer())

        '''
        test_preds = self.sess.run(self.yhat, feed_dict = {self.X: X})
        test_perf = self.get_performance(test_preds, Y)

        if task_name == None:
            task_name = task_id

        print(" [*] Evaluation, Task:%s, test_acc: %.4f" % (str(task_name), test_perf))
        return test_perf
        '''
        test_acc = self.sess.run(self.accuracy, feed_dict={self.X: X, self.Y: Y})
        return test_acc

    def prediction(self, X):
        preds = self.sess.run(self.yhat, feed_dict = {self.X: X})
        return preds

    def destroy_graph(self):
        tf.reset_default_graph()

    def avg_sparsity(self, task_id):
        n_params, zeros = 0, 0
        for idx in range(self.n_layers):
            with tf.variable_scope("layer%d"%(idx+1), reuse = True):
                if idx < (self.n_layers-1):
                    w = tf.get_variable('weight')
                else:
                    w = tf.get_variable('weight_%d'%task_id)
            m_value = self.sess.run(w)
            size = 1.
            shape = m_value.shape
            for dim in shape:
                size = size * dim
            n_params += size
            nzero = float(np.count_nonzero(m_value))
            zeros += (size - nzero)
        return (zeros+1) / (n_params+1)
