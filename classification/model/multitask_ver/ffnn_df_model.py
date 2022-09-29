import numpy as np
import tensorflow as tf

from utils.utils import *
from utils.utils_nn import *
from utils.utils_df_nn import *


########################################################
####   ELLA Neural Network for Multi-task Learning  ####
####         linear relation btw KB and TS          ####
########################################################
## data_list = [train_x, train_y, valid_x, valid_y, test_x, test_y]
##           or [ [(trainx_1, trainy_1), ..., (trainx_t, trainy_t)], [(validx_1, validy_1), ...], [(testx_1, testy_1), ...] ]
## dim_knokw_base = [KB0, KB1, ...]
class ELLA_FFNN_linear_relation_minibatch():
    def __init__(self, num_tasks, dim_layers, dim_know_base, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, classification=False):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
        self.KB_size = dim_know_base
        self.l1_reg_scale = reg_scale[0]
        self.l2_reg_scale = reg_scale[1]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### data
        self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)

        ### define regularizer
        l1_reg, l2_reg = tf.contrib.layers.l1_regularizer(scale=self.l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.l2_reg_scale)

        #### mini-batch for training/validation/test data
        train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data(self.data, self.batch_size, self.data_index, [-1]+self.image_size, do_MTL=True, num_tasks=self.num_tasks)

        #### layers of model for train data

        self.train_models, self.KB_param, self.TS_param = [], [], []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer(train_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            elif layer_cnt == self.num_layers-1:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, para_activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            else:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            self.train_models.append(layer_tmp)
            self.KB_param = self.KB_param + KB_para_tmp
            self.TS_param = self.TS_param + TS_para_tmp
        #self.train_eval = self.train_models[-1]
        self.param = [self.KB_param, self.TS_param]

        #### layers of model for validation data
        self.valid_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(valid_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            self.valid_models.append(layer_tmp)
        #self.valid_eval = self.valid_models[-1]

        #### layers of model for test data
        self.test_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(test_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            self.test_models.append(layer_tmp)
        #self.test_eval = self.test_models[-1]

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term1, reg_term2 = tf.contrib.layers.apply_regularization(l1_reg, reg_var), tf.contrib.layers.apply_regularization(l2_reg, reg_var)

        if classification and self.layers_size[-1]>1:
            self.train_eval = [tf.nn.softmax(self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.valid_eval = [tf.nn.softmax(self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_eval = [tf.nn.softmax(self.test_models[-1][x]) for x in range(self.num_tasks)]

            self.train_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=train_y_batch[x], logits=self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]
            self.valid_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=valid_y_batch[x], logits=self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=test_y_batch[x], logits=self.test_models[-1][x]) for x in range(self.num_tasks)]
        else:
            self.train_eval = [self.train_models[-1][x] for x in range(self.num_tasks)]
            self.valid_eval = [self.valid_models[-1][x] for x in range(self.num_tasks)]
            self.test_eval = [self.test_models[-1][x] for x in range(self.num_tasks)]

            self.train_loss = [2.0* tf.nn.l2_loss(self.train_eval[x]-train_y_batch[x]) for x in range(self.num_tasks)]
            self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]
            self.valid_loss = [2.0* tf.nn.l2_loss(self.valid_eval[x]-valid_y_batch[x]) for x in range(self.num_tasks)]
            self.test_loss = [2.0* tf.nn.l2_loss(self.test_eval[x]-test_y_batch[x]) for x in range(self.num_tasks)]

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        #### functions only for classification problem
        if classification and self.layers_size[-1]<2:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.train_eval[x]>0.5), (train_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.valid_eval[x]>0.5), (valid_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.test_eval[x]>0.5), (test_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]

        elif classification:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.train_eval[x], 1), tf.argmax(train_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.valid_eval[x], 1), tf.argmax(valid_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.test_eval[x], 1), tf.argmax(test_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]


class ELLA_FFNN_linear_relation_minibatch2():
    def __init__(self, num_tasks, dim_layers, dim_know_base, dim_task_specific, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, classification=False):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
        self.KB_size = dim_know_base
        self.TS_size = dim_task_specific
        self.l1_reg_scale = reg_scale[0]
        self.l2_reg_scale = reg_scale[1]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### data
        self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)

        ### define regularizer
        l1_reg, l2_reg = tf.contrib.layers.l1_regularizer(scale=self.l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.l2_reg_scale)

        #### mini-batch for training/validation/test data
        train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data(self.data, self.batch_size, self.data_index, [-1]+self.image_size, do_MTL=True, num_tasks=self.num_tasks)

        #### layers of model for train data
        self.train_models, self.KB_param, self.TS_param = [], [], []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer2(train_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            elif layer_cnt == self.num_layers-1:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer2(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            else:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer2(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            self.train_models.append(layer_tmp)
            self.KB_param = self.KB_param + KB_para_tmp
            self.TS_param = self.TS_param + TS_para_tmp
        #self.train_eval = self.train_models[-1]
        self.param = [self.KB_param, self.TS_param]

        #### layers of model for validation data
        self.valid_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(valid_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            self.valid_models.append(layer_tmp)
        #self.valid_eval = self.valid_models[-1]

        #### layers of model for test data
        self.test_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(test_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, para_activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            self.test_models.append(layer_tmp)
        #self.test_eval = self.test_models[-1]

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term1, reg_term2 = tf.contrib.layers.apply_regularization(l1_reg, reg_var), tf.contrib.layers.apply_regularization(l2_reg, reg_var)

        if classification and self.layers_size[-1]>1:
            self.train_eval = [tf.nn.softmax(self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.valid_eval = [tf.nn.softmax(self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_eval = [tf.nn.softmax(self.test_models[-1][x]) for x in range(self.num_tasks)]

            self.train_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=train_y_batch[x], logits=self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]
            self.valid_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=valid_y_batch[x], logits=self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=test_y_batch[x], logits=self.test_models[-1][x]) for x in range(self.num_tasks)]
        else:
            self.train_eval = [self.train_models[-1][x] for x in range(self.num_tasks)]
            self.valid_eval = [self.valid_models[-1][x] for x in range(self.num_tasks)]
            self.test_eval = [self.test_models[-1][x] for x in range(self.num_tasks)]

            self.train_loss = [2.0* tf.nn.l2_loss(self.train_eval[x]-train_y_batch[x]) for x in range(self.num_tasks)]
            self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]
            self.valid_loss = [2.0* tf.nn.l2_loss(self.valid_eval[x]-valid_y_batch[x]) for x in range(self.num_tasks)]
            self.test_loss = [2.0* tf.nn.l2_loss(self.test_eval[x]-test_y_batch[x]) for x in range(self.num_tasks)]

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        #### functions only for classification problem
        if classification and self.layers_size[-1]<2:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.train_eval[x]>0.5), (train_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.valid_eval[x]>0.5), (valid_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.test_eval[x]>0.5), (test_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]

        elif classification:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.train_eval[x], 1), tf.argmax(train_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.valid_eval[x], 1), tf.argmax(valid_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.test_eval[x], 1), tf.argmax(test_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]





########################################################
####   ELLA Neural Network for Multi-task Learning  ####
####       nonlinear relation btw KB and TS         ####
########################################################
## data_list = [train_x, train_y, valid_x, valid_y, test_x, test_y]
##           or [ [(trainx_1, trainy_1), ..., (trainx_t, trainy_t)], [(validx_1, validy_1), ...], [(testx_1, testy_1), ...] ]
## dim_knokw_base = [KB0, KB1, ...]
class ELLA_FFNN_nonlinear_relation_minibatch():
    def __init__(self, num_tasks, dim_layers, dim_know_base, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, classification=False):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
        self.KB_size = dim_know_base
        self.l1_reg_scale = reg_scale[0]
        self.l2_reg_scale = reg_scale[1]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### data
        self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)

        ### define regularizer
        l1_reg, l2_reg = tf.contrib.layers.l1_regularizer(scale=self.l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.l2_reg_scale)

        #### mini-batch for training/validation/test data
        train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data(self.data, self.batch_size, self.data_index, [-1]+self.image_size, do_MTL=True, num_tasks=self.num_tasks)

        #### layers of model for train data
        self.train_models, self.KB_param, self.TS_param = [], [], []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer(train_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            elif layer_cnt == self.num_layers-1:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            else:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            self.train_models.append(layer_tmp)
            self.KB_param = self.KB_param + KB_para_tmp
            self.TS_param = self.TS_param + TS_para_tmp
        #self.train_eval = self.train_models[-1]
        self.param = [self.KB_param, self.TS_param]

        #### layers of model for validation data
        self.valid_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(valid_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            self.valid_models.append(layer_tmp)
        #self.valid_eval = self.valid_models[-1]

        #### layers of model for test data
        self.test_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(test_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.num_tasks, layer_cnt, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[2*self.num_tasks*layer_cnt:2*self.num_tasks*(layer_cnt+1)])
            self.test_models.append(layer_tmp)
        #self.test_eval = self.test_models[-1]

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term1, reg_term2 = tf.contrib.layers.apply_regularization(l1_reg, reg_var), tf.contrib.layers.apply_regularization(l2_reg, reg_var)

        if classification and self.layers_size[-1]>1:
            self.train_eval = [tf.nn.softmax(self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.valid_eval = [tf.nn.softmax(self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_eval = [tf.nn.softmax(self.test_models[-1][x]) for x in range(self.num_tasks)]

            self.train_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=train_y_batch[x], logits=self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]
            self.valid_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=valid_y_batch[x], logits=self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=test_y_batch[x], logits=self.test_models[-1][x]) for x in range(self.num_tasks)]
        else:
            self.train_eval = [self.train_models[-1][x] for x in range(self.num_tasks)]
            self.valid_eval = [self.valid_models[-1][x] for x in range(self.num_tasks)]
            self.test_eval = [self.test_models[-1][x] for x in range(self.num_tasks)]

            self.train_loss = [2.0* tf.nn.l2_loss(self.train_eval[x]-train_y_batch[x]) for x in range(self.num_tasks)]
            self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]
            self.valid_loss = [2.0* tf.nn.l2_loss(self.valid_eval[x]-valid_y_batch[x]) for x in range(self.num_tasks)]
            self.test_loss = [2.0* tf.nn.l2_loss(self.test_eval[x]-test_y_batch[x]) for x in range(self.num_tasks)]

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        #### functions only for classification problem
        if classification and self.layers_size[-1]<2:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.train_eval[x]>0.5), (train_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.valid_eval[x]>0.5), (valid_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.test_eval[x]>0.5), (test_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]

        elif classification:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.train_eval[x], 1), tf.argmax(train_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.valid_eval[x], 1), tf.argmax(valid_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.test_eval[x], 1), tf.argmax(test_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]







class ELLA_FFNN_nonlinear_relation_minibatch2():
    def __init__(self, num_tasks, dim_layers, dim_know_base, dim_task_specific, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, classification=False):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
        self.KB_size = dim_know_base
        self.TS_size = dim_task_specific
        self.l1_reg_scale = reg_scale[0]
        self.l2_reg_scale = reg_scale[1]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### data
        self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)

        ### define regularizer
        l1_reg, l2_reg = tf.contrib.layers.l1_regularizer(scale=self.l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.l2_reg_scale)

        #### mini-batch for training/validation/test data
        train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data(self.data, self.batch_size, self.data_index, [-1]+self.image_size, do_MTL=True, num_tasks=self.num_tasks)

        #### layers of model for train data
        self.train_models, self.KB_param, self.TS_param = [], [], []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer2(train_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            elif layer_cnt == self.num_layers-1:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer2(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            else:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_nonlinear_relation_layer2(self.train_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, KB_reg_type=l2_reg, TS_reg_type=l2_reg)
            self.train_models.append(layer_tmp)
            self.KB_param = self.KB_param + KB_para_tmp
            self.TS_param = self.TS_param + TS_para_tmp
        #self.train_eval = self.train_models[-1]
        self.param = [self.KB_param, self.TS_param]

        #### layers of model for validation data
        self.valid_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(valid_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(self.valid_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            self.valid_models.append(layer_tmp)
        #self.valid_eval = self.valid_models[-1]

        #### layers of model for test data
        self.test_models = []
        for layer_cnt in range(self.num_layers):
            if layer_cnt == 0:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(test_x_batch, self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            elif layer_cnt == self.num_layers-1:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, activation_fn=None, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            else:
                layer_tmp, _, _ = new_ELLA_nonlinear_relation_layer2(self.test_models[layer_cnt-1], self.layers_size[layer_cnt], self.layers_size[layer_cnt+1], self.KB_size[layer_cnt], self.TS_size[layer_cnt], self.num_tasks, layer_cnt, KB_param=self.KB_param[layer_cnt], TS_param=self.TS_param[4*self.num_tasks*layer_cnt:4*self.num_tasks*(layer_cnt+1)])
            self.test_models.append(layer_tmp)
        #self.test_eval = self.test_models[-1]

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term1, reg_term2 = tf.contrib.layers.apply_regularization(l1_reg, reg_var), tf.contrib.layers.apply_regularization(l2_reg, reg_var)

        if classification and self.layers_size[-1]>1:
            self.train_eval = [tf.nn.softmax(self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.valid_eval = [tf.nn.softmax(self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_eval = [tf.nn.softmax(self.test_models[-1][x]) for x in range(self.num_tasks)]

            self.train_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=train_y_batch[x], logits=self.train_models[-1][x]) for x in range(self.num_tasks)]
            self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]
            self.valid_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=valid_y_batch[x], logits=self.valid_models[-1][x]) for x in range(self.num_tasks)]
            self.test_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=test_y_batch[x], logits=self.test_models[-1][x]) for x in range(self.num_tasks)]
        else:
            self.train_eval = [self.train_models[-1][x] for x in range(self.num_tasks)]
            self.valid_eval = [self.valid_models[-1][x] for x in range(self.num_tasks)]
            self.test_eval = [self.test_models[-1][x] for x in range(self.num_tasks)]

            self.train_loss = [2.0* tf.nn.l2_loss(self.train_eval[x]-train_y_batch[x]) for x in range(self.num_tasks)]
            self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]
            self.valid_loss = [2.0* tf.nn.l2_loss(self.valid_eval[x]-valid_y_batch[x]) for x in range(self.num_tasks)]
            self.test_loss = [2.0* tf.nn.l2_loss(self.test_eval[x]-test_y_batch[x]) for x in range(self.num_tasks)]

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]

        #### functions only for classification problem
        if classification and self.layers_size[-1]<2:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.train_eval[x]>0.5), (train_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.valid_eval[x]>0.5), (valid_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal((self.test_eval[x]>0.5), (test_y_batch[x]>0.5)), tf.float32)) for x in range(self.num_tasks)]

        elif classification:
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.train_eval[x], 1), tf.argmax(train_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.valid_eval[x], 1), tf.argmax(valid_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.test_eval[x], 1), tf.argmax(test_y_batch[x], 1)), tf.float32)) for x in range(self.num_tasks)]














###################################################################################
###################################################################################
#### new trial!!


class ELLA_FFNN_nonlinear_tensordot_relation_minibatch():
    def __init__(self, num_tasks, dim_layers, dim_know_base, batch_size, learning_rate, learning_rate_decay, reg_scale, data_list, relation_activation_fn=tf.nn.relu, classification=False):
        self.num_tasks = num_tasks
        self.num_layers = len(dim_layers)-1
        self.layers_size = dim_layers
        self.KB_size = dim_know_base
        self.num_KB_para = self.num_layers
        self.num_TS_para = 0
        self.l1_reg_scale = reg_scale[0]
        self.l2_reg_scale = reg_scale[1]
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        #### data
        self.data = tflized_data(data_list, do_MTL=True, num_tasks=self.num_tasks)

        #### placeholder of model
        self.data_index = tf.placeholder(dtype=tf.int32)
        self.epoch = tf.placeholder(dtype=tf.float32)

        ### define regularizer
        l1_reg, l2_reg = tf.contrib.layers.l1_regularizer(scale=self.l1_reg_scale), tf.contrib.layers.l2_regularizer(scale=self.l2_reg_scale)

        #### mini-batch for training/validation/test data
        train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_data(self.data, self.batch_size, self.data_index, do_MTL=True, num_tasks=self.num_tasks)

        if classification:
            output_type='classification'
        else:
            output_type=None

        #### layers of model for train data
        self.train_models = []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.KB_param, self.TS_param = new_ELLA_fc_tensordot_net(train_x_batch[task_cnt], self.layers_size, self.KB_size, para_activation_fn=relation_activation_fn, KB_reg_type=l2_reg, TS_reg_type=l2_reg, output_type=output_type, task_index=task_cnt)
                self.num_TS_para = len(self.TS_param)
            else:
                model_tmp, _, TS_param_tmp = new_ELLA_fc_tensordot_net(train_x_batch[task_cnt], self.layers_size, self.KB_size, para_activation_fn=relation_activation_fn, KB_params=self.KB_param, TS_reg_type=l2_reg, output_type=output_type, task_index=task_cnt)
                self.TS_param = self.TS_param + TS_param_tmp
            self.train_models.append(model_tmp)
        self.param = self.KB_param + self.TS_param

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _ = new_ELLA_fc_tensordot_net(valid_x_batch[task_cnt], self.layers_size, self.KB_size, para_activation_fn=relation_activation_fn, KB_params=self.KB_param, TS_params=self.TS_param[task_cnt*self.num_TS_para:], output_type=output_type, task_index=task_cnt)
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _ = new_ELLA_fc_tensordot_net(test_x_batch[task_cnt], self.layers_size, self.KB_size, para_activation_fn=relation_activation_fn, KB_params=self.KB_param, TS_params=self.TS_param[task_cnt*self.num_TS_para:], output_type=output_type, task_index=task_cnt)
            self.test_models.append(model_tmp)

        #### loss functions of model
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term1, reg_term2 = tf.contrib.layers.apply_regularization(l1_reg, reg_var), tf.contrib.layers.apply_regularization(l2_reg, reg_var)

        self.train_eval, self.valid_eval, self.test_eval, self.train_loss, self.valid_loss, self.test_loss, self.train_accuracy, self.valid_accuracy, self.test_accuracy = mtl_model_output_functions([self.train_models, self.valid_models, self.test_models], [train_y_batch, valid_y_batch, test_y_batch], self.num_tasks, self.layers_size[-1], classification=classification)

        self.train_loss_and_reg = [self.train_loss[x] + reg_term1 + reg_term2 for x in range(self.num_tasks)]

        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss_and_reg[x]) for x in range(self.num_tasks)]
