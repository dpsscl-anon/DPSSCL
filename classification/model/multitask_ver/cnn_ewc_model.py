import numpy as np
import tensorflow as tf

from utils.utils import *
from utils.utils_nn import *


def gen_lagged_vars_holder(list_of_tf_vars):
    lagged_list = []
    for var in list_of_tf_vars:
        lagged_list.append(tf.Variable(tf.constant(0.0, shape=var.get_shape().as_list()), trainable=False))
    return lagged_list

def clone_variable_list(variable_list):
    return [tf.identity(var) for var in variable_list]

def clone_variable_list_by_shape(variable_list, trainable):
    return [tf.Variable(tf.constant(0.0, shape=var.get_shape().as_list()), trainable=trainable) for var in variable_list]

def convert_weight_bias_to_cnn_fc_params(weights, biases):
    assert (len(weights) == len(biases)), "Given weights and biases have different number of elements"
    cnn_params, fc_params = [], []
    for (w, b) in zip(weights, biases):
        w_shape = w.get_shape().as_list()
        if len(w_shape) > 2:
            cnn_params += [w, b]
        else:
            fc_params += [w, b]
    return cnn_params, fc_params

def unaggregated_nll(x_fisher, y_fisher, ewc_batch_size, net_param, weights, biases, skip_connections):
    x_fisher_shape = x_fisher.get_shape().as_list()
    assert ( (x_fisher_shape[0]==ewc_batch_size) and (y_fisher.get_shape().as_list()[0]==ewc_batch_size) ), "Size mis-match between data and ewc_batch_size"
    x_examples, y_examples = tf.unstack(x_fisher), tf.unstack(y_fisher)
    weights_per_example = [clone_variable_list(weights) for _ in range(0, ewc_batch_size)]
    biases_per_example = [clone_variable_list(biases) for _ in range(0, ewc_batch_size)]
    nll_list = []
    for (x, y, biases, weights) in zip(x_examples, y_examples, biases_per_example, weights_per_example):
        cnn_params, fc_params = convert_weight_bias_to_cnn_fc_params(weights, biases)
        temp, _, _ = new_cnn_fc_net(tf.reshape(x, [1]+x_fisher_shape[1:]), net_param[0], net_param[1], net_param[2], net_param[3], cnn_params=cnn_params, fc_params=fc_params, padding_type=net_param[4], max_pool=net_param[5], pool_sizes=net_param[6], dropout=False, input_size=net_param[7], output_type=net_param[8], skip_connections=skip_connections)
        scores = temp[-1]
        nll = - tf.reduce_sum(y * tf.nn.log_softmax(scores))
        nll_list.append(nll)
    nlls = tf.stack(nll_list)
    return tf.reduce_sum(nlls), biases_per_example, weights_per_example

def fisher_minibatch_sum(nll_per_example, biases_per_example, weights_per_example):
    bias_grads_per_example = [tf.gradients(nll_per_example, biases) for biases in biases_per_example]
    weight_grads_per_example = [tf.gradients(nll_per_example, weights) for weights in weights_per_example]
    return sum_of_squared_gradients(bias_grads_per_example, weight_grads_per_example)

def sum_of_squared_gradients(bias_grads_per_example, weight_grads_per_example):
    num_layers = len(bias_grads_per_example[0])
    bias_grads2_sum = []
    weight_grads2_sum = []
    for layer in range(num_layers):
        bias_grad2_sum = tf.add_n([tf.square(example[layer]) for example in bias_grads_per_example])
        weight_grad2_sum = tf.add_n([tf.square(example[layer]) for example in weight_grads_per_example])
        bias_grads2_sum.append(bias_grad2_sum)
        weight_grads2_sum.append(weight_grad2_sum)
    return bias_grads2_sum + weight_grads2_sum



#### Convolutional & Fully-connected Neural Net
class MTL_CNN_EWC_minibatch():
    def __init__(self, num_tasks, dim_channels, dim_fcs, input_size, dim_kernel, dim_strides, batch_size, learning_rate, learning_rate_decay, data_list, fisher_multiplier, ewc_batch_size=100, padding_type='SAME', max_pooling=False, dim_pool=None, dropout=False, saveData=False, skip_connect=[]):
        self.num_tasks = num_tasks
        #### num_layers == len(channels_size) + len(fc_size) && len(fc_size) >= 1
        self.num_layers = len(dim_channels) + len(dim_fcs)
        self.cnn_channels_size = [input_size[-1]]+dim_channels    ## include dim of input channel
        self.num_outputs = dim_fcs[-1]

        self.cnn_kernel_size = dim_kernel     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = dim_strides
        self.pool_size = dim_pool      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = dim_fcs
        self.input_size = input_size    ## img_width * img_height * img_channel
        self.learn_rate = learning_rate
        self.learn_rate_decay = learning_rate_decay
        self.batch_size = batch_size

        self.ewc_batch_size = ewc_batch_size
        self.fisher_multiplier = fisher_multiplier

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
            self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.num_outputs]) for _ in range(self.num_tasks)]
            #### mini-batch for training/validation/test data
            train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch = minibatched_cnn_data_not_in_gpu(self.model_input, self.true_output, [-1] + self.input_size, do_MTL=True, num_tasks=self.num_tasks)

        #### layers of model for train data
        self.train_models = []
        for task_cnt in range(self.num_tasks):
            if task_cnt == 0:
                model_tmp, self.cnn_param, self.fc_param = new_cnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=None, fc_params=None, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', skip_connections=skip_connect)
            else:
                model_tmp, _, _ = new_cnn_fc_net(train_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', skip_connections=skip_connect)
            self.train_models.append(model_tmp)
        self.param = self.cnn_param + self.fc_param

        #### layers of model for validation data
        self.valid_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _ = new_cnn_fc_net(valid_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', skip_connections=skip_connect)
            self.valid_models.append(model_tmp)

        #### layers of model for test data
        self.test_models = []
        for task_cnt in range(self.num_tasks):
            model_tmp, _, _ = new_cnn_fc_net(test_x_batch[task_cnt], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, cnn_params=self.cnn_param, fc_params=self.fc_param, padding_type=padding_type, max_pool=max_pooling, pool_sizes=dim_pool, dropout=dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], output_type='classification', skip_connections=skip_connect)
            self.test_models.append(model_tmp)

        #### methods for evaluation metrics
        with tf.name_scope('Model_Eval'):
            self.train_eval = [tf.nn.softmax(self.train_models[x][-1]) for x in range(num_tasks)]
            self.valid_eval = [tf.nn.softmax(self.valid_models[x][-1]) for x in range(num_tasks)]
            self.test_eval = [tf.nn.softmax(self.test_models[x][-1]) for x in range(num_tasks)]

            self.train_output_label = [tf.argmax(self.train_models[x][-1], 1) for x in range(num_tasks)]
            self.valid_output_label = [tf.argmax(self.valid_models[x][-1], 1) for x in range(num_tasks)]
            self.test_output_label = [tf.argmax(self.test_models[x][-1], 1) for x in range(num_tasks)]

        ## Model_Loss is defined below because EWC requires unique loss

        with tf.name_scope('Model_Accuracy'):
            self.train_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.train_models[x][-1], 1), tf.argmax(train_y_batch[x], 1)), tf.float32)) for x in range(num_tasks)]
            self.valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.valid_models[x][-1], 1), tf.argmax(valid_y_batch[x], 1)), tf.float32)) for x in range(num_tasks)]
            self.test_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.test_models[x][-1], 1), tf.argmax(test_y_batch[x], 1)), tf.float32)) for x in range(num_tasks)]

        #### Set-up for Elastic Weight Consolidation (EWC)
        ## placeholders for Fisher Loss
        self.x_fisher = tf.placeholder(dtype=tf.float32, shape=[self.ewc_batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]])
        x_fisher_reshaped = tf.reshape(self.x_fisher, [-1]+self.input_size)
        self.y_fisher = tf.placeholder(dtype=tf.float32, shape=[self.ewc_batch_size, self.num_outputs])
        self.ewc_batches = tf.placeholder(dtype=tf.float32)

        ## compute approx. Fisher Matrix
        net_param = [self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, self.fc_size, padding_type, max_pooling, dim_pool, self.input_size[0:2], 'classification']
        weights, biases = self.param[0::2], self.param[1::2]
        theta = biases+weights

        nll, biases_per_example, weights_per_example = unaggregated_nll(x_fisher_reshaped, self.y_fisher, self.ewc_batch_size, net_param, weights, biases, skip_connections=skip_connect)
        self.fisher_minibatch = fisher_minibatch_sum(nll, biases_per_example, weights_per_example)

        ## fisher op
        self.fisher_diagonal = clone_variable_list_by_shape(biases, trainable=False) + clone_variable_list_by_shape(weights, trainable=False)

        self.fisher_accumulate_op = [tf.assign_add(f1, f2) for f1, f2 in zip(self.fisher_diagonal, self.fisher_minibatch)]
        self.fisher_full_batch_average_op = [tf.assign(var, var*(1.0/(self.ewc_batches*float(self.ewc_batch_size)))) for var in self.fisher_diagonal]
        self.fisher_zero_op = [tf.assign(tensor, tf.zeros_like(tensor)) for tensor in self.fisher_diagonal]

        ## lagged parameters
        self.lagged_param = gen_lagged_vars_holder(theta)
        self.update_lagged_param_op = [v1.assign(v2) for v1, v2 in zip(self.lagged_param, theta)]

        with tf.name_scope('Model_Loss'):
            ## EWC use
            self.train_loss = [tf.nn.softmax_cross_entropy_with_logits(labels=train_y_batch[x], logits=self.train_models[x][-1]) for x in range(num_tasks)]
            self.fisher_penalty = tf.add_n([tf.reduce_sum(tf.square(w1-w2)*f) for w1, w2, f in zip(theta, self.lagged_param, self.fisher_diagonal)])

            self.valid_loss = self.test_loss = [None for x in range(num_tasks)]

        #### functions of model
        self.update = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x]) for x in range(self.num_tasks)]
        self.update_ewc = [tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)).minimize(self.train_loss[x]+(fisher_multiplier/2)*self.fisher_penalty) for x in range(self.num_tasks)]

        self.num_trainable_var = count_trainable_var()

    def update_lagged_param(self, sess):
        sess.run(self.update_lagged_param_op)

    def update_fisher_full_batch(self, sess, train_x, train_y):
        num_data = train_x.shape[0]
        ewc_batches = num_data // self.ewc_batch_size

        sess.run(self.fisher_zero_op)
        for ewc_batch_cnt in range(ewc_batches):
            batch_x, batch_y = train_x[ewc_batch_cnt*self.ewc_batch_size:(ewc_batch_cnt+1)*self.ewc_batch_size, :], train_y[ewc_batch_cnt*self.ewc_batch_size:(ewc_batch_cnt+1)*self.ewc_batch_size, :]
            self.accumulate_fisher(sess, batch_x, batch_y)
        sess.run(self.fisher_full_batch_average_op, {self.ewc_batches: ewc_batches})
        #self.update_lagged_param(sess)

    def accumulate_fisher(self, sess, batch_xs, batch_ys):
        sess.run(self.fisher_accumulate_op, feed_dict={self.x_fisher: batch_xs, self.y_fisher: batch_ys})