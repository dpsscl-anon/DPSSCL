import numpy as np
import tensorflow as tf

from utils.utils import new_bias
from utils.utils_nn import new_fc_layer, new_fc_net, new_cnn_layer, new_cnn_net
from utils.utils_df_nn import new_ELLA_KB_param
from utils.utils_tensor_factorization import TensorProducer


#### function to generate HPS model with all dense layers
def new_hardparam_fc_fc_nets(net_inputs, hid_sizes_shared, hid_sizes_specific, num_task, activation_fn=tf.nn.relu, shared_params=None, specific_params=None, output_type=None):
    num_acc_specific_params, num_specific_params_tmp = [0], 0
    for a in hid_sizes_specific:
        num_specific_params_tmp += 2*len(a)
        num_acc_specific_params.append(num_specific_params_tmp)

    hps_models, shared_params_return, specific_params_return = [], [], []
    for task_cnt in range(num_task):
        #### generate network common to tasks
        if task_cnt == 0:
            shared_model_tmp, shared_params_return = new_fc_net(net_inputs[task_cnt], hid_sizes_shared, activation_fn=activation_fn, params=shared_params, output_type='same')
        else:
            shared_model_tmp, _ = new_fc_net(net_inputs[task_cnt], hid_sizes_shared, activation_fn=activation_fn, params=shared_params_return, output_type='same')

        #### generate task-dependent network
        if specific_params is None:
            specific_model_tmp, specific_params_tmp = new_fc_net(shared_model_tmp[-1], hid_sizes_specific[task_cnt], activation_fn=activation_fn, params=None, output_type=output_type)
        else:
            specific_model_tmp, specific_params_tmp = new_fc_net(shared_model_tmp[-1], hid_sizes_specific[task_cnt], activation_fn=activation_fn, params=specific_params[num_acc_specific_params[task_cnt]:num_acc_specific_params[task_cnt+1]], output_type=output_type)

        hps_models.append(shared_model_tmp+specific_model_tmp)
        specific_params_return = specific_params_return + specific_params_tmp

    return (hps_models, shared_params_return, specific_params_return)



#################################################################################
#### functions for Multi-task Tensor Factorized model

######## Multi-task Learning - based on Yongxin Yang and Timothy Hospedales, Deep Multi-task Representation Learning: A Tensor Factorisation Approach
#### function to generate parameters of tensor factored convolutional layer
def new_tensorfactored_weight(shape, num_task, factor_type='Tucker', factor_eps_or_k=0.01, init_val=None):
    if init_val is None:
        if len(shape) == 2:
            W_init = np.random.rand(shape[0], shape[1], num_task)
        elif len(shape) == 4:
            W_init = np.random.rand(shape[0], shape[1], shape[2], shape[3], num_task)
        else:
            return (None, None)
    else:
        W_init = init_val

    W_tmp, W_dict = TensorProducer(W_init, factor_type, eps_or_k=factor_eps_or_k, return_true_var=True)

    if len(shape) == 2:
        W = [W_tmp[:, :, i] for i in range(num_task)]
    elif len(shape) == 4:
        W = [W_tmp[:, :, :, :, i] for i in range(num_task)]
    return (W, W_dict)


def new_tensorfactored_fc_weights(hid_sizes, num_task, factor_type='Tucker', factor_eps_or_k=0.01):
    num_layers = len(hid_sizes)-1
    param_tmp = [[] for i in range(num_task)]
    for layer_cnt in range(num_layers):
        W_tmp, _ = new_tensorfactored_weight(hid_sizes[layer_cnt:layer_cnt+2], num_task, factor_type, factor_eps_or_k)
        bias_tmp = [new_bias(shape=[hid_sizes[layer_cnt+1]]) for i in range(num_task)]
        for task_cnt in range(num_task):
            param_tmp[task_cnt].append(W_tmp[task_cnt])
            param_tmp[task_cnt].append(bias_tmp[task_cnt])

    param = []
    for task_cnt in range(num_task):
        param = param + param_tmp[task_cnt]
    return param


def new_tensorfactored_fc_nets(net_inputs, hid_sizes, num_task, activation_fn=tf.nn.relu, params=None, factor_type='Tucker', factor_eps_or_k=0.01, output_type=None):
    num_para_per_model = 2*(len(hid_sizes)-1)

    with tf.name_scope('TF_fc_net'):
        if len(hid_sizes) < 2:
            #### for the case that hard-parameter shared network does not have shared layers
            return (net_inputs, [])
        elif params is None:
            #### network & parameters are new
            params = new_tensorfactored_fc_weights(hid_sizes, num_task, factor_type, factor_eps_or_k)
            fc_models = []
            for task_cnt in range(num_task):
                fc_model_tmp, _ = new_fc_net(net_inputs[task_cnt], hid_sizes[1:], activation_fn=activation_fn, params=params[task_cnt*num_para_per_model:(task_cnt+1)*num_para_per_model], output_type=output_type)
                fc_models.append(fc_model_tmp)
        else:
            #### network generated from existing parameters
            fc_models = []
            for task_cnt in range(num_task):
                fc_model_tmp, _ = new_fc_net(net_inputs[task_cnt], hid_sizes[1:], activation_fn=activation_fn, params=params[task_cnt*num_para_per_model:(task_cnt+1)*num_para_per_model], output_type=output_type)
                fc_models.append(fc_model_tmp)

    return (fc_models, params)


def new_tensorfactored_fc_fc_nets(net_inputs, hid_sizes_shared, hid_sizes_specific, num_task, activation_fn=tf.nn.relu, shared_params=None, specific_params=None, factor_type='Tucker', factor_eps_or_k=0.01, output_type=None):
    tf_shared_net, tf_shared_param = new_tensorfactored_fc_nets(net_inputs, hid_sizes_shared, num_task, activation_fn, shared_params, factor_type, factor_eps_or_k, output_type='same')

    num_acc_specific_fc_params, num_specific_fc_params_tmp = [0], 0
    for a in hid_sizes_specific:
        num_specific_fc_params_tmp += 2*len(a)
        num_acc_specific_fc_params.append(num_specific_fc_params_tmp)

    overall_net, tf_specific_param = [], []
    for task_cnt in range(num_task):
        #n_in = int(tf_shared_net[task_cnt][-1].shape[1])
        if specific_params is None:
            fc_net_tmp, fc_param_tmp = new_fc_net(tf_shared_net[task_cnt][-1], hid_sizes_specific[task_cnt], activation_fn=activation_fn, output_type=output_type)
        else:
            fc_net_tmp, fc_param_tmp = new_fc_net(tf_shared_net[task_cnt][-1], hid_sizes_specific[task_cnt], activation_fn=activation_fn, params=specific_params[num_acc_specific_fc_params[task_cnt]:num_acc_specific_fc_params[task_cnt+1]], output_type=output_type)
        overall_net.append(tf_shared_net[task_cnt]+fc_net_tmp)
        tf_specific_param = tf_specific_param + fc_param_tmp

    return (overall_net, tf_shared_param, tf_specific_param)


def new_tensorfactored_cnn_weights(k_sizes, ch_sizes, num_task, factor_type='Tucker', factor_eps_or_k=0.01, init_params=None):
    num_layers = len(ch_sizes)-1
    if init_params is None:
        init_params = [None for _ in range(num_layers)]

    param_tmp = [[] for i in range(num_task)]
    for layer_cnt in range(num_layers):
        W_tmp, _ = new_tensorfactored_weight(k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], num_task, factor_type, factor_eps_or_k, init_params[layer_cnt])
        bias_tmp = [new_bias(shape=[ch_sizes[layer_cnt+1]]) for i in range(num_task)]
        for task_cnt in range(num_task):
            param_tmp[task_cnt].append(W_tmp[task_cnt])
            param_tmp[task_cnt].append(bias_tmp[task_cnt])

    param = []
    for task_cnt in range(num_task):
        param = param + param_tmp[task_cnt]
    return param


def new_tensorfactored_cnn_nets(net_inputs, k_sizes, ch_sizes, stride_sizes, num_task, activation_fn=tf.nn.relu, params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, input_size=[0, 0], factor_type='Tucker', factor_eps_or_k=0.01, init_params=None, skip_connections=[]):
    num_para_per_model = len(k_sizes)

    with tf.name_scope('TF_conv_net'):
        if len(k_sizes) < 2:
            #### for the case that hard-parameter shared network does not have shared layers
            return (net_inputs, [])
        elif params is None:
            #### network & parameters are new
            params = new_tensorfactored_cnn_weights(k_sizes, ch_sizes, num_task, factor_type, factor_eps_or_k, init_params)

            # params
            cnn_models = []
            for task_cnt in range(num_task):
                cnn_model_tmp, _, output_dim = new_cnn_net(net_inputs[task_cnt], k_sizes, ch_sizes, stride_sizes, activation_fn=activation_fn, params=params[task_cnt*num_para_per_model:(task_cnt+1)*num_para_per_model], padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=flat_output, input_size=input_size, skip_connections=list(skip_connections))
                cnn_models.append(cnn_model_tmp)
        else:
            #### network generated from existing parameters
            cnn_models = []
            for task_cnt in range(num_task):
                cnn_model_tmp, _, output_dim = new_cnn_net(net_inputs[task_cnt], k_sizes, ch_sizes, stride_sizes, activation_fn=activation_fn, params=params[task_cnt*num_para_per_model:(task_cnt+1)*num_para_per_model], padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=flat_output, input_size=input_size, skip_connections=list(skip_connections))
                cnn_models.append(cnn_model_tmp)
    return (cnn_models, params, output_dim)

def new_tensorfactored_cnn_fc_nets(net_inputs, k_sizes, ch_sizes, stride_sizes, fc_sizes, num_task, cnn_activation_fn=tf.nn.relu, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], factor_type='Tucker', factor_eps_or_k=0.01, output_type=None, init_param=None, skip_connections=[]):
    num_acc_specific_params, num_specific_params_tmp = [0], 0
    for a in fc_sizes:
        num_specific_params_tmp += 2 * len(a)
        num_acc_specific_params.append(num_specific_params_tmp)

    num_cnn_layers = len(k_sizes)//2
    assert ((init_param is None) or ( (init_param is not None) and (len(init_param)==2*num_cnn_layers*num_task+num_acc_specific_params[-1]) )), "Given initializing parameter doesn't match to the size of architecture"
    if init_param is None:
        layerwise_cnn_init_params = [None for _ in range(num_cnn_layers)]
    else:
        layerwise_cnn_init_params = []
        for cnn_layer_cnt in range(num_cnn_layers):
            layerwise_cnn_param_tmp = []
            for task_cnt in range(num_task):
                layerwise_cnn_param_tmp.append(init_param[2*(cnn_layer_cnt+num_cnn_layers*task_cnt)+num_acc_specific_params[task_cnt]])
            layerwise_cnn_param = np.transpose(np.stack(layerwise_cnn_param_tmp), axes=[1, 2, 3, 4, 0])
            layerwise_cnn_init_params.append(layerwise_cnn_param)

    if cnn_params is None:
        cnn_models, cnn_params, output_dim = new_tensorfactored_cnn_nets(net_inputs, k_sizes, ch_sizes, stride_sizes, num_task, activation_fn=cnn_activation_fn, params=cnn_params, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, input_size=input_size, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k, init_params=layerwise_cnn_init_params, skip_connections=skip_connections)
    else:
        cnn_models, _, output_dim = new_tensorfactored_cnn_nets(net_inputs, k_sizes, ch_sizes, stride_sizes, num_task, activation_fn=cnn_activation_fn, params=cnn_params, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, input_size=input_size, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k, skip_connections=skip_connections)

    fc_models = []
    if fc_params is None:
        fc_params = []
        for task_cnt in range(num_task):
            fc_model_tmp, fc_param_tmp = new_fc_net(cnn_models[task_cnt][-1], fc_sizes[task_cnt], activation_fn=fc_activation_fn, params=None, output_type=output_type)
            fc_models.append(fc_model_tmp)
            fc_params = fc_params + fc_param_tmp
    else:
        for task_cnt in range(num_task):
            fc_model_tmp, _ = new_fc_net(cnn_models[task_cnt][-1], fc_sizes[task_cnt], activation_fn=fc_activation_fn, params=fc_params[num_acc_specific_params[task_cnt]:num_acc_specific_params[task_cnt+1]], output_type=output_type)
            fc_models.append(fc_model_tmp)

    models = []
    for task_cnt in range(num_task):
        models.append(cnn_models[task_cnt]+fc_models[task_cnt])
    return (models, cnn_params, fc_params)


def new_tensorfactored_cnn_tensorfactored_fc_nets(net_inputs, k_sizes, ch_sizes, stride_sizes, fc_sizes, num_task, cnn_activation_fn=tf.nn.relu, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], factor_type='Tucker', factor_eps_or_k=0.01, output_type=None):
    if cnn_params is None:
        cnn_models, cnn_params, output_dim = new_tensorfactored_cnn_nets(net_inputs, k_sizes, ch_sizes, stride_sizes, num_task, activation_fn=cnn_activation_fn, params=cnn_params, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, input_size=input_size, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k)
    else:
        cnn_models, _, output_dim = new_tensorfactored_cnn_nets(net_inputs, k_sizes, ch_sizes, stride_sizes, num_task, activation_fn=cnn_activation_fn, params=cnn_params, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, input_size=input_size, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k)

    fc_sizes = output_dim+fc_sizes
    cnn_models_last_layers = [cnn_models[x][-1] for x in range(num_task)]
    if fc_params is None:
        fc_models, fc_params = new_tensorfactored_fc_nets(cnn_models_last_layers, fc_sizes, num_task, activation_fn=fc_activation_fn, params=fc_params, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k, output_type=output_type)
    else:
        fc_models, _ = new_tensorfactored_fc_nets(cnn_models_last_layers, fc_sizes, num_task, activation_fn=fc_activation_fn, params=fc_params, factor_type=factor_type, factor_eps_or_k=factor_eps_or_k, output_type=output_type)

    models = []
    for task_cnt in range(num_task):
        models.append(cnn_models[task_cnt]+fc_models[task_cnt])
    return (models, cnn_params, fc_params)


#################################################################################

#### Dynamic Filter Network
#### function to generate knowledge-base parameters for ELLA_simple layer
def new_KB_simple_param(dim_kb, input_dim, output_dim, layer_number, reg_type):
    w_name, b_name = 'KB_W'+str(layer_number), 'KB_b'+str(layer_number)
    return [tf.get_variable(name=w_name, shape=[dim_kb[0], input_dim, output_dim], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=b_name, shape=[dim_kb[1], output_dim], dtype=tf.float32, regularizer=reg_type)]

#### function to generate task-specific parameters for ELLA_simple layer
def new_TS_simple_param(dim_kb, layer_number, task_number, reg_type):
    sw_name, sb_name = 'TS_W'+str(layer_number)+'_'+str(task_number), 'TS_b'+str(layer_number)+'_'+str(task_number)
    return [tf.get_variable(name=sw_name, shape=[1, dim_kb[0]], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=sb_name, shape=[1, dim_kb[1]], dtype=tf.float32, regularizer=reg_type)]


#### function to generate knowledge-base parameters for ELLA_tensorfactor layer
def new_KB_nonlinear_relation_param(dim_kb, layer_number, reg_type):
    kb_name = 'KB_'+str(layer_number)
    return tf.get_variable(name=kb_name, shape=[1, dim_kb], dtype=tf.float32, regularizer=reg_type)

#### function to generate task-specific parameters for ELLA_tensorfactor layer
def new_TS_nonlinear_relation_param(dim_kb, input_dim, output_dim, layer_number, task_number, reg_type):
    ts_w_name, ts_b_name = 'TS_W'+str(layer_number)+'_'+str(task_number), 'TS_b'+str(layer_number)+'_'+str(task_number)
    return [tf.get_variable(name=ts_w_name, shape=[dim_kb, (input_dim+1)*output_dim], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_b_name, shape=[(input_dim+1)*output_dim], dtype=tf.float32, regularizer=reg_type)]

#### function to generate task-specific parameters for ELLA_tensorfactor layer
def new_TS_nonlinear_relation_param2(dim_kb, dim_ts, input_dim, output_dim, layer_number, task_number, reg_type):
    ts_w_name, ts_b_name, ts_k_name, ts_p_name = 'TS_W'+str(layer_number)+'_'+str(task_number), 'TS_b'+str(layer_number)+'_'+str(task_number), 'TS_K'+str(layer_number)+'_'+str(task_number), 'TS_p'+str(layer_number)+'_'+str(task_number)
    return [tf.get_variable(name=ts_w_name, shape=[dim_kb, dim_ts], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_b_name, shape=[dim_ts], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_k_name, shape=[dim_ts, (input_dim+1)*output_dim], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_p_name, shape=[(input_dim+1)*output_dim], dtype=tf.float32, regularizer=reg_type)]

#### function to generate task-specific parameters for ELLA_tensorfactor layer
def new_ELLA_TS_param(shape, layer_number, task_number, reg_type):
    #ts_w_name, ts_b_name, ts_k_name, ts_p_name = 'TS_W0_'+str(layer_number)+'_'+str(task_number), 'TS_b0_'+str(layer_number)+'_'+str(task_number), 'TS_W1_'+str(layer_number)+'_'+str(task_number), 'TS_b1_'+str(layer_number)+'_'+str(task_number)
    #return [tf.get_variable(name=ts_w_name, shape=shape[0], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_b_name, shape=shape[1], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_k_name, shape=shape[2], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_p_name, shape=shape[3], dtype=tf.float32, regularizer=reg_type)]
    return [tf.get_variable(name='TS_W'+str(x)+'_'+str(layer_number)+'_'+str(task_number), shape=shape[x], dtype=tf.float32, regularizer=reg_type) for x in range(len(shape))]

#### function to generate task-specific parameters for ELLA_tensorfactor layer
def new_ELLA_cnn_deconv_tensordot_TS_param2_reshape(shape, layer_number, task_number, reg_type):
    ts_w_name, ts_b_name, ts_k_name, ts_k_name2, ts_p_name = 'TS_DeconvW0_'+str(layer_number)+'_'+str(task_number), 'TS_Deconvb0_'+str(layer_number)+'_'+str(task_number), 'TS_tdot_W1_'+str(layer_number)+'_'+str(task_number), 'TS_tdot_W2_'+str(layer_number)+'_'+str(task_number), 'TS_tdot_b0_'+str(layer_number)+'_'+str(task_number)
    return [tf.get_variable(name=ts_w_name, shape=shape[0], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_b_name, shape=shape[1], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_k_name, shape=shape[2], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_k_name2, shape=shape[3], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_p_name, shape=shape[4], dtype=tf.float32, regularizer=reg_type)]




############################################
####      functions for ELLA-FFNN       ####
####  nonlinear relation btw KB and TS  ####
############################################
#### function to add ELLA_tensorfactor layer
#def new_ELLA_nonlinear_relation_layer(layer_input_list, input_dim, output_dim, KB_dim, num_task, layer_number, activation_fn=tf.nn.relu, para_activation_fn=tf.tanh, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None):
def new_ELLA_nonlinear_relation_layer(layer_input_list, input_dim, output_dim, KB_dim, num_task, layer_number, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None):
    if KB_param is None:
        KB_param = new_KB_nonlinear_relation_param(KB_dim, layer_number, KB_reg_type)
    if TS_param is None:
        TS_param = []
        for cnt in range(num_task):
            TS_param = TS_param + new_TS_nonlinear_relation_param(KB_dim, input_dim, output_dim, layer_number, cnt, TS_reg_type)

    layer_eqn = []
    for task_cnt in range(num_task):
        if para_activation_fn is None:
            para_tmp = tf.matmul(KB_param, TS_param[2*task_cnt]) + TS_param[2*task_cnt+1]
        else:
            para_tmp = para_activation_fn(tf.matmul(KB_param, TS_param[2*task_cnt]) + TS_param[2*task_cnt+1])

        W_tmp, b = tf.split(tf.reshape(para_tmp, [(input_dim+1)*output_dim]), [input_dim*output_dim, output_dim])
        W = tf.reshape(W_tmp, [input_dim, output_dim])

        if activation_fn is None:
            layer_eqn.append( tf.matmul(layer_input_list[task_cnt], W)+b )
        else:
            layer_eqn.append( activation_fn( tf.matmul(layer_input_list[task_cnt], W)+b ) )
    return layer_eqn, [KB_param], TS_param


#### function to add ELLA_tensorfactor layer
def new_ELLA_nonlinear_relation_layer2(layer_input_list, input_dim, output_dim, KB_dim, TS_dim, num_task, layer_number, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None):
    if KB_param is None:
        KB_param = new_KB_nonlinear_relation_param(KB_dim, layer_number, KB_reg_type)
    if TS_param is None:
        TS_param = []
        for cnt in range(num_task):
            TS_param = TS_param + new_TS_nonlinear_relation_param2(KB_dim, TS_dim, input_dim, output_dim, layer_number, cnt, TS_reg_type)

    layer_eqn = []
    for task_cnt in range(num_task):
        if para_activation_fn is None:
            para_tmp = tf.matmul(tf.matmul(KB_param, TS_param[4*task_cnt]) + TS_param[4*task_cnt+1], TS_param[4*task_cnt+2]) + TS_param[4*task_cnt+3]
        else:
            para_tmp = tf.matmul(para_activation_fn(tf.matmul(KB_param, TS_param[4*task_cnt]) + TS_param[4*task_cnt+1]), TS_param[4*task_cnt+2]) + TS_param[4*task_cnt+3]

        W_tmp, b = tf.split(tf.reshape(para_tmp, [(input_dim+1)*output_dim]), [input_dim*output_dim, output_dim])
        W = tf.reshape(W_tmp, [input_dim, output_dim])

        if activation_fn is None:
            layer_eqn.append( tf.matmul(layer_input_list[task_cnt], W)+b )
        else:
            layer_eqn.append( activation_fn( tf.matmul(layer_input_list[task_cnt], W)+b ) )
    return layer_eqn, [KB_param], TS_param


############################################################
#####   functions for adding ELLA network (CNN ver)    #####
############################################################
#### function to generate convolutional layer with shared knowledge base
def new_ELLA_cnn_layer(layer_input, k_size, ch_size, stride_size, KB_size, layer_num, task_num, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_size=None, skip_connect_input=None):
    with tf.name_scope('ELLA_conv_shared_KB'):
        if KB_param is None:
            ## KB \in R^{h \times c}
            KB_param = new_ELLA_KB_param(KB_size, layer_num, task_num, KB_reg_type)
        if TS_param is None:
            ## TS1 \in R^{(H*W*Ch_in+1) \times h}
            ## TS2 \in R^{c \times Ch_out}
            ## tensordot(KB, TS1) -> R^{c \times (H*W*Ch_in+1)}
            ## tensordot(..., TS2) -> R^{(H*W*Ch_in+1) \times Ch_out}
            TS_param = new_ELLA_TS_param([[k_size[0]*k_size[1]*ch_size[0]+1, KB_size[0]], [1, k_size[0]*k_size[1]*ch_size[0]+1], [KB_size[1], ch_size[1]], [1, ch_size[1]]], layer_num, task_num, TS_reg_type)

    with tf.name_scope('ELLA_conv_TS'):
        if para_activation_fn is None:
            para_tmp = tf.add(tf.tensordot(KB_param, TS_param[0], [[0], [1]]), TS_param[1])
        else:
            para_tmp = para_activation_fn(tf.add(tf.tensordot(KB_param, TS_param[0], [[0], [1]]), TS_param[1]))
        para_last = tf.add(tf.tensordot(para_tmp, TS_param[2], [[0], [0]]), TS_param[3])

        W_tmp, b = tf.split(tf.reshape(para_last, [(k_size[0]*k_size[1]*ch_size[0]+1)*ch_size[1]]), [k_size[0]*k_size[1]*ch_size[0]*ch_size[1], ch_size[1]])
        W = tf.reshape(W_tmp, [k_size[0], k_size[1], ch_size[0], ch_size[1]])

    layer_eqn, _ = new_cnn_layer(layer_input, k_size+ch_size, stride_size=stride_size, activation_fn=activation_fn, weight=W, bias=b, padding_type=padding_type, max_pooling=max_pool, pool_size=pool_size, skip_connect_input=skip_connect_input)
    return layer_eqn, [KB_param], TS_param


#### function to generate network of convolutional layers with shared knowledge base
def new_ELLA_cnn_net(net_input, k_sizes, ch_sizes, stride_sizes, KB_sizes, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_params=None, TS_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, input_size=[0, 0], task_index=0, skip_connections=[]):
    _num_TS_param_per_layer = 4

    ## first element : make new KB&TS / second element : make new TS / third element : not make new para
    control_flag = [(KB_params is None and TS_params is None), (not (KB_params is None) and (TS_params is None)), not (KB_params is None or TS_params is None)]
    if control_flag[1]:
        TS_params = []
    elif control_flag[0]:
        KB_params, TS_params = [], []

    layers_for_skip, next_skip_connect = [net_input], None
    with tf.name_scope('ELLA_conv_net'):
        layers = []
        for layer_cnt in range(len(k_sizes)//2):
            next_skip_connect = skip_connections.pop(0) if (len(skip_connections) > 0 and next_skip_connect is None) else next_skip_connect
            if next_skip_connect is not None:
                skip_connect_in, skip_connect_out = next_skip_connect
                assert (skip_connect_in > -1 and skip_connect_out > -1), "Given skip connection has error (try connecting non-existing layer)"
            else:
                skip_connect_in, skip_connect_out = -1, -1

            if layer_cnt == skip_connect_out:
                processed_skip_connect_input = layers_for_skip[skip_connect_in]
                for layer_cnt_tmp in range(skip_connect_in, skip_connect_out):
                    if max_pool and (pool_sizes[2*layer_cnt_tmp]>1 or pool_sizes[2*layer_cnt_tmp+1]>1):
                        processed_skip_connect_input = tf.nn.max_pool(processed_skip_connect_input, ksize=[1]+pool_sizes[2*layer_cnt_tmp:2*(layer_cnt_tmp+1)]+[1], strides=[1]+pool_sizes[2*layer_cnt_tmp:2*(layer_cnt_tmp+1)]+[1], padding=padding_type)
            else:
                processed_skip_connect_input = None

            if layer_cnt == 0 and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_cnn_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif layer_cnt == 0 and control_flag[1]:
                layer_tmp, _, TS_para_tmp = new_ELLA_cnn_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif layer_cnt == 0 and control_flag[2]:
                layer_tmp, _, _ = new_ELLA_cnn_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_cnn_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[1]:
                layer_tmp, _, TS_para_tmp = new_ELLA_cnn_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[2]:
                layer_tmp, _, _ = new_ELLA_cnn_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)

            layers.append(layer_tmp)
            layers_for_skip.append(layer_tmp)
            if control_flag[1]:
                TS_params = TS_params + TS_para_tmp
            elif control_flag[0]:
                KB_params = KB_params + KB_para_tmp
                TS_params = TS_params + TS_para_tmp
            if layer_cnt == skip_connect_out:
                next_skip_connect = None

        #### flattening output
        if flat_output:
            output_dim = [int(layers[-1].shape[1]*layers[-1].shape[2]*layers[-1].shape[3])]
            layers.append(tf.reshape(layers[-1], [-1, output_dim[0]]))
        else:
            output_dim = layers[-1].shape[1:]

        #### add dropout layer
        if dropout:
            layers.append(tf.nn.dropout(layers[-1], dropout_prob))
    return (layers, KB_params, TS_params, output_dim)


############################################################
#####   functions for adding ELLA network (FFNN ver)   #####
############################################################
#### function to generate fully connected layer with shared knowledge base
def new_ELLA_fc_layer(layer_input, input_dim, output_dim, KB_dim, layer_num, task_num, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None):
    with tf.name_scope('ELLA_fc_shared_KB'):
        if KB_param is None:
            ## KB \in R^{1 \times h}
            KB_param = new_ELLA_KB_param([1, KB_dim[0]], layer_num, task_num, KB_reg_type)
        if TS_param is None:
            ## TS1 \in R^{h \times c}
            ## TS2 \in R^{c \times ((Cin+1)*Cout)}
            TS_param = new_ELLA_TS_param([KB_dim, [1, KB_dim[1]], [KB_dim[1], (input_dim+1)*output_dim], [1, (input_dim+1)*output_dim]], layer_num, task_num, TS_reg_type)

    with tf.name_scope('ELLA_fc_TS'):
        if para_activation_fn is None:
            para_tmp = tf.add(tf.matmul(KB_param, TS_param[0]), TS_param[1])
        else:
            para_tmp = para_activation_fn(tf.add(tf.matmul(KB_param, TS_param[0]), TS_param[1]))
        para_last = tf.add(tf.matmul(para_tmp, TS_param[2]), TS_param[3])

        W_tmp, b = tf.split(tf.reshape(para_last, [(input_dim+1)*output_dim]), [input_dim*output_dim, output_dim])
        W = tf.reshape(W_tmp, [input_dim, output_dim])

    layer_eqn, _ = new_fc_layer(layer_input, output_dim, activation_fn=activation_fn, weight=W, bias=b)
    return layer_eqn, [KB_param], TS_param


#### function to generate network of fully connected layers with shared knowledge base
def new_ELLA_fc_net(net_input, dim_layers, dim_KBs, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_params=None, TS_params=None, KB_reg_type=None, TS_reg_type=None, output_type=None, layer_start_index=0, task_index=0):
    _num_TS_param_per_layer = 4

    ## first element : make new KB&TS / second element : make new TS / third element : not make new para
    control_flag = [(KB_params is None and TS_params is None), (not (KB_params is None) and (TS_params is None)), not (KB_params is None or TS_params is None)]
    if control_flag[1]:
        TS_params = []
    elif control_flag[0]:
        KB_params, TS_params = [], []

    with tf.name_scope('ELLA_fc_net'):
        layers = []
        for layer_cnt in range(len(dim_layers)-1):
            if layer_cnt == 0 and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_fc_layer(net_input, dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == 0 and control_flag[1]:
                layer_tmp, _, TS_para_tmp = new_ELLA_fc_layer(net_input, dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == 0 and control_flag[2]:
                layer_tmp, _, _ = new_ELLA_fc_layer(net_input, dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == len(dim_layers)-2 and output_type is 'classification' and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn='classification', para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == len(dim_layers)-2 and output_type is 'classification' and control_flag[1]:
                layer_tmp, _, TS_para_tmp = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn='classification', para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == len(dim_layers)-2 and output_type is 'classification' and control_flag[2]:
                layer_tmp, _, _ = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn='classification', para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == len(dim_layers)-2 and output_type is None and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=None, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == len(dim_layers)-2 and output_type is None and control_flag[1]:
                layer_tmp, _, TS_para_tmp = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=None, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == len(dim_layers)-2 and output_type is None and control_flag[2]:
                layer_tmp, _, _ = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=None, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif control_flag[1]:
                layer_tmp, _, TS_para_tmp = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif control_flag[2]:
                layer_tmp, _, _ = new_ELLA_fc_layer(layers[layer_cnt-1], dim_layers[layer_cnt], dim_layers[layer_cnt+1], dim_KBs[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)

            layers.append(layer_tmp)
            if control_flag[1]:
                TS_params = TS_params + TS_para_tmp
            elif control_flag[0]:
                KB_params = KB_params + KB_para_tmp
                TS_params = TS_params + TS_para_tmp
    return (layers, KB_params, TS_params)


#### function to generate network of cnn->ffnn
def new_ELLA_cnn_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_KB_sizes, fc_KB_sizes, cnn_activation_fn=tf.nn.relu, cnn_para_activation_fn=tf.nn.relu, cnn_KB_params=None, cnn_TS_params=None, fc_activation_fn=tf.nn.relu, fc_para_activation_fn=tf.nn.relu, fc_KB_params=None, fc_TS_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, task_index=0, skip_connections=[]):
    ## add CNN layers
    cnn_model, cnn_KB_params, cnn_TS_params, cnn_output_dim = new_ELLA_cnn_net(net_input, k_sizes, ch_sizes, stride_sizes, cnn_KB_sizes, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_params=cnn_KB_params, TS_params=cnn_TS_params, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, input_size=input_size, task_index=task_index, skip_connections=skip_connections)

    ## add fc layers
    fc_model, fc_KB_params, fc_TS_params = new_ELLA_fc_net(cnn_model[-1], [cnn_output_dim[0]]+fc_sizes, fc_KB_sizes, activation_fn=fc_activation_fn, para_activation_fn=fc_para_activation_fn, KB_params=fc_KB_params, TS_params=fc_TS_params, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, output_type=output_type, layer_start_index=len(k_sizes)//2, task_index=task_index)

    return (cnn_model+fc_model, cnn_KB_params, cnn_TS_params, fc_KB_params, fc_TS_params)




#### function to generate fully-connected layer with shared knowledge base
#### KB \in R^{h \times w}
#### TS1 : TS W1 \in R^{(hidden_{i-1}+1) \times h}
#### TS2 : TS W1 \in R^{w \times hidden_{i}}
#### [W; b] = activation(TS1 * KB) * TS2
def new_ELLA_fc_tensordot_layer(layer_input, layer_size, KB_size, layer_num, task_num, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None):
    with tf.name_scope('ELLA_ffnn_KBTS'):
        if KB_param is None:
            KB_param = new_ELLA_KB_param(KB_size, layer_num, task_num, KB_reg_type)
        if TS_param is None:
            TS_param = new_ELLA_TS_param([[layer_size[0]+1, KB_size[0]], [1, KB_size[1]], [KB_size[1], layer_size[1]]], layer_num, task_num, TS_reg_type)

    with tf.name_scope('ELLA_ffnn_param'):
        para_tmp = tf.add(tf.tensordot(TS_param[0], KB_param, [[1], [0]]), TS_param[1])
        if para_activation_fn is not None:
            para_tmp = para_activation_fn(para_tmp)
        para_tmp = tf.tensordot(para_tmp, TS_param[2], [[1], [0]])
        W, b = tf.split(para_tmp, [layer_size[0], 1], 0)

    layer_eqn, _ = new_fc_layer(layer_input, layer_size[1], activation_fn=activation_fn, weight=W, bias=b)
    return layer_eqn, [KB_param], TS_param


def new_ELLA_fc_tensordot_net(net_input, layer_sizes, KB_sizes, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_params=None, TS_params=None, KB_reg_type=None, TS_reg_type=None, output_type=None, layer_start_index=0, task_index=0):
    _num_TS_param_per_layer = 3

    ## first element : make new KB&TS / second element : make new TS / third element : not make new para
    control_flag = [(KB_params is None and TS_params is None), (not (KB_params is None) and (TS_params is None)), not (KB_params is None or TS_params is None)]
    if control_flag[1]:
        TS_params = []
    elif control_flag[0]:
        KB_params, TS_params = [], []

    with tf.name_scope('ELLA_fc_net'):
        layers = []
        for layer_cnt in range(len(layer_sizes)-1):
            if layer_cnt == 0 and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_fc_tensordot_layer(net_input, layer_sizes[layer_cnt:layer_cnt+2], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == 0 and control_flag[1]:
                layer_tmp, _, TS_para_tmp = new_ELLA_fc_tensordot_layer(net_input, layer_sizes[layer_cnt:layer_cnt+2], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == 0 and control_flag[2]:
                layer_tmp, _, _ = new_ELLA_fc_tensordot_layer(net_input, layer_sizes[layer_cnt:layer_cnt+2], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == len(layer_sizes)-2 and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_fc_tensordot_layer(layers[layer_cnt-1], layer_sizes[layer_cnt:layer_cnt+2], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=output_type, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == len(layer_sizes)-2 and control_flag[1]:
                layer_tmp, _, TS_para_tmp = new_ELLA_fc_tensordot_layer(layers[layer_cnt-1], layer_sizes[layer_cnt:layer_cnt+2], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=output_type, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif layer_cnt == len(layer_sizes)-2 and control_flag[2]:
                layer_tmp, _, _ = new_ELLA_fc_tensordot_layer(layers[layer_cnt-1], layer_sizes[layer_cnt:layer_cnt+2], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=output_type, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp = new_ELLA_fc_tensordot_layer(layers[layer_cnt-1], layer_sizes[layer_cnt:layer_cnt+2], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif control_flag[1]:
                layer_tmp, _, TS_para_tmp = new_ELLA_fc_tensordot_layer(layers[layer_cnt-1], layer_sizes[layer_cnt:layer_cnt+2], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)
            elif control_flag[2]:
                layer_tmp, _, _ = new_ELLA_fc_tensordot_layer(layers[layer_cnt-1], layer_sizes[layer_cnt:layer_cnt+2], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_start_index+layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type)

            layers.append(layer_tmp)
            if control_flag[1]:
                TS_params = TS_params + TS_para_tmp
            elif control_flag[0]:
                KB_params = KB_params + KB_para_tmp
                TS_params = TS_params + TS_para_tmp
    return (layers, KB_params, TS_params)


#### KB_size : [filter_height(and width), num_of_channel]
#### TS_size : [deconv_filter_height(and width), deconv_filter_channel0, deconv_filter_channel1]
#### TS_stride_size : [stride_in_height, stride_in_width]
def new_ELLA_cnn_deconv_tensordot_layer2_reshape(layer_input, k_size, ch_size, stride_size, KB_size, TS_size, TS_stride_size, layer_num, task_num, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_size=None, skip_connect_input=None):
    assert (k_size[0] == k_size[1] and k_size[0] == (KB_size[0]-1)*TS_stride_size[0]+1), "CNN kernel size does not match the output size of Deconv from KB"

    with tf.name_scope('ELLA_cdnn_KB'):
        if KB_param is None:
            ## KB \in R^{h \times w \times c}
            KB_param = new_ELLA_KB_param([1, KB_size[0], KB_size[0], KB_size[1]], layer_num, task_num, KB_reg_type)
        if TS_param is None:
            ## TS0 : Deconv W \in R^{h \times w \times kb_c_out \times (ts_c0 * ts_c1)}
            ## TS1 : Deconv bias \in R^{(ts_c0 * ts_c1)}
            ## TS2 : tensor W \in R^{ts_c0 \times ch_in}
            ## TS3 : tensor W \in R^{ts_c1 \times ch_out}
            ## TS4 : Conv bias \in R^{ch_out}
            TS_param = new_ELLA_cnn_deconv_tensordot_TS_param2_reshape([[TS_size[0], TS_size[0], TS_size[1]*TS_size[2], KB_size[1]], [1, 1, 1, TS_size[1]*TS_size[2]], [TS_size[1], ch_size[0]], [TS_size[2], ch_size[1]], [1, 1, 1, ch_size[1]]], layer_num, task_num, TS_reg_type)

    with tf.name_scope('ELLA_cdnn_TS'):
        para_tmp = tf.add(tf.nn.conv2d_transpose(KB_param, TS_param[0], [1, k_size[0], k_size[1], TS_size[1]*TS_size[2]], strides=[1, TS_stride_size[0], TS_stride_size[1], 1]), TS_param[1])
        para_tmp = tf.reshape(para_tmp, [k_size[0], k_size[1], TS_size[1], TS_size[2]])
        if para_activation_fn is not None:
            para_tmp = para_activation_fn(para_tmp)
        para_tmp = tf.tensordot(para_tmp, TS_param[2], [[2], [0]])
        W = tf.tensordot(para_tmp, TS_param[3], [[2], [0]])
        b = TS_param[4]

    layer_eqn, _ = new_cnn_layer(layer_input, k_size+ch_size, stride_size=stride_size, activation_fn=activation_fn, weight=W, bias=b, padding_type=padding_type, max_pooling=max_pool, pool_size=pool_size, skip_connect_input=skip_connect_input)
    return layer_eqn, [KB_param], TS_param, [W, b]


#### function to generate network of convolutional layers with shared knowledge base
def new_ELLA_cnn_deconv_tensordot_net2_reshape(net_input, k_sizes, ch_sizes, stride_sizes, KB_sizes, TS_sizes, TS_stride_sizes, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_params=None, TS_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, input_size=[0, 0], task_index=0, skip_connections=[]):
    _num_TS_param_per_layer = 5

    ## first element : make new KB&TS / second element : make new TS / third element : not make new para / fourth element : make new KB
    control_flag = [(KB_params is None and TS_params is None), (not (KB_params is None) and (TS_params is None)), not (KB_params is None or TS_params is None), ((KB_params is None) and not (TS_params is None))]
    if control_flag[1]:
        TS_params = []
    elif control_flag[3]:
        KB_params = []
    elif control_flag[0]:
        KB_params, TS_params = [], []
    cnn_gen_params = []

    layers_for_skip, next_skip_connect = [net_input], None
    with tf.name_scope('ELLA_cdnn_net'):
        layers = []
        for layer_cnt in range(len(k_sizes)//2):
            next_skip_connect = skip_connections.pop(0) if (len(skip_connections) > 0 and next_skip_connect is None) else next_skip_connect
            if next_skip_connect is not None:
                skip_connect_in, skip_connect_out = next_skip_connect
                assert (skip_connect_in > -1 and skip_connect_out > -1), "Given skip connection has error (try connecting non-existing layer)"
            else:
                skip_connect_in, skip_connect_out = -1, -1

            if layer_cnt == skip_connect_out:
                processed_skip_connect_input = layers_for_skip[skip_connect_in]
                for layer_cnt_tmp in range(skip_connect_in, skip_connect_out):
                    if max_pool and (pool_sizes[2*layer_cnt_tmp]>1 or pool_sizes[2*layer_cnt_tmp+1]>1):
                        processed_skip_connect_input = tf.nn.max_pool(processed_skip_connect_input, ksize=[1]+pool_sizes[2*layer_cnt_tmp:2*(layer_cnt_tmp+1)]+[1], strides=[1]+pool_sizes[2*layer_cnt_tmp:2*(layer_cnt_tmp+1)]+[1], padding=padding_type)
            else:
                processed_skip_connect_input = None

            if layer_cnt == 0 and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_ELLA_cnn_deconv_tensordot_layer2_reshape(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif layer_cnt == 0 and control_flag[1]:
                layer_tmp, _, TS_para_tmp, cnn_gen_para_tmp = new_ELLA_cnn_deconv_tensordot_layer2_reshape(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif layer_cnt == 0 and control_flag[2]:
                layer_tmp, _, _, cnn_gen_para_tmp = new_ELLA_cnn_deconv_tensordot_layer2_reshape(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif layer_cnt == 0 and control_flag[3]:
                layer_tmp, KB_para_tmp, _, cnn_gen_para_tmp = new_ELLA_cnn_deconv_tensordot_layer2_reshape(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_ELLA_cnn_deconv_tensordot_layer2_reshape(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[1]:
                layer_tmp, _, TS_para_tmp, cnn_gen_para_tmp = new_ELLA_cnn_deconv_tensordot_layer2_reshape(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[2]:
                layer_tmp, _, _, cnn_gen_para_tmp = new_ELLA_cnn_deconv_tensordot_layer2_reshape(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[3]:
                layer_tmp, KB_para_tmp, _, cnn_gen_para_tmp = new_ELLA_cnn_deconv_tensordot_layer2_reshape(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)

            layers.append(layer_tmp)
            layers_for_skip.append(layer_tmp)
            cnn_gen_params = cnn_gen_params + cnn_gen_para_tmp
            if control_flag[1]:
                TS_params = TS_params + TS_para_tmp
            elif control_flag[3]:
                KB_params = KB_params + KB_para_tmp
            elif control_flag[0]:
                KB_params = KB_params + KB_para_tmp
                TS_params = TS_params + TS_para_tmp
            if layer_cnt == skip_connect_out:
                next_skip_connect = None

        #### flattening output
        if flat_output:
            output_dim = [int(layers[-1].shape[1]*layers[-1].shape[2]*layers[-1].shape[3])]
            layers.append(tf.reshape(layers[-1], [-1, output_dim[0]]))
        else:
            output_dim = layers[-1].shape[1:]

        #### add dropout layer
        if dropout:
            layers.append(tf.nn.dropout(layers[-1], dropout_prob))
    return (layers, KB_params, TS_params, cnn_gen_params, output_dim)


#### function to generate network of cnn (with shared KB through deconv)-> simple ffnn
def new_ELLA_cnn_deconv_tensordot_fc_net2_reshape(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, cnn_activation_fn=tf.nn.relu, cnn_para_activation_fn=tf.nn.relu, cnn_KB_params=None, cnn_TS_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, task_index=0, skip_connections=[]):
    ## add CNN layers
    cnn_model, cnn_KB_params, cnn_TS_params, cnn_gen_params, cnn_output_dim = new_ELLA_cnn_deconv_tensordot_net2_reshape(net_input, k_sizes, ch_sizes, stride_sizes, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_params=cnn_KB_params, TS_params=cnn_TS_params, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, input_size=input_size, task_index=task_index, skip_connections=skip_connections)

    ## add fc layers
    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, tensorboard_name_scope='fc_net')

    return (cnn_model+fc_model, cnn_KB_params, cnn_TS_params, cnn_gen_params, fc_params)


#############################################################################################################
####  function for Conv-FC nets whose conv layers are freely set to shared across tasks by DeconvFactor  ####
####          this class can learn which layers to be shared and which ones to be task-specific          ####
#############################################################################################################
def new_Hybrid_DFCNN_net_auto_sharing(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, sharing_params, cnn_activation_fn=tf.nn.relu, cnn_para_activation_fn=tf.nn.relu, cnn_KB_params=None, cnn_TS_params=None, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, task_index=0, skip_connections=[], highway_connect_type=0, cnn_highway_params=None, sharing_var_scale=1.0, trainable=True, trainable_KB=True, mixture_type='sum', layer_types=None):
    _num_TS_param_per_layer = 4
    num_conv_layers = [len(k_sizes)//2, len(ch_sizes)-1, len(stride_sizes)//2, len(sharing_params), len(cnn_KB_sizes)//2, len(cnn_TS_sizes)//2, len(cnn_TS_stride_sizes)//2]
    assert (all([(num_conv_layers[i]==num_conv_layers[i+1]) for i in range(len(num_conv_layers)-1)])), "Parameters related to conv layers are wrong!"
    num_conv_layers = num_conv_layers[0]
    if layer_types is None:
        layer_types = [0.0 for _ in range(num_conv_layers)]
    else:
        assert (len(layer_types) == num_conv_layers), "Given type of each layer is wrong!"

    ## add CNN layers
    cnn_sharing = [tf.Variable(v, trainable=True, name='Conv_Sharing%d'%(l), dtype=tf.float32, constraint=lambda t: tf.clip_by_value(t, -sharing_var_scale, sharing_var_scale)) if (l_type>-0.5 and l_type<0.5) else tf.constant(l_type, dtype=tf.float32) for l, (v, l_type) in enumerate(zip(sharing_params, layer_types))]

    ## first element : make new KB&TS / second element : make new TS / third element : not make new para / fourth element : make new KB
    control_flag = [(cnn_KB_params is None and cnn_TS_params is None), (not (cnn_KB_params is None) and (cnn_TS_params is None)), not (cnn_KB_params is None or cnn_TS_params is None), ((cnn_KB_params is None) and not (cnn_TS_params is None))]
    if control_flag[1]:
        cnn_TS_params = []
    elif control_flag[3]:
        cnn_KB_params = []
    elif control_flag[0]:
        cnn_KB_params, cnn_TS_params = [], []
    cnn_gen_params = []

    if cnn_params is None:
        cnn_params = [None for _ in range(2*num_conv_layers)]

    layers_for_skip, next_skip_connect = [net_input], None
    with tf.name_scope('Hybrid_DFCNN_auto_sharing'):
        cnn_model, cnn_params_to_return, cnn_highway_params_to_return = [], [], []
        cnn_KB_to_return, cnn_TS_to_return = [], []
        for layer_cnt in range(num_conv_layers):
            KB_para_tmp, TS_para_tmp, para_tmp = [None], [None for _ in range(_num_TS_param_per_layer)], [None, None]
            highway_para_tmp = [None, None] if cnn_highway_params is None else cnn_highway_params[2*layer_cnt:2*(layer_cnt+1)]
            cnn_gen_para_tmp = [None, None]

            next_skip_connect = skip_connections.pop(0) if (len(skip_connections) > 0 and next_skip_connect is None) else next_skip_connect
            if next_skip_connect is not None:
                skip_connect_in, skip_connect_out = next_skip_connect
                assert (skip_connect_in > -1 and skip_connect_out > -1), "Given skip connection has error (try connecting non-existing layer)"
            else:
                skip_connect_in, skip_connect_out = -1, -1

            if layer_cnt == skip_connect_out:
                processed_skip_connect_input = layers_for_skip[skip_connect_in]
                for layer_cnt_tmp in range(skip_connect_in, skip_connect_out):
                    if max_pool and (pool_sizes[2*layer_cnt_tmp]>1 or pool_sizes[2*layer_cnt_tmp+1]>1):
                        processed_skip_connect_input = tf.nn.max_pool(processed_skip_connect_input, ksize=[1]+pool_sizes[2*layer_cnt_tmp:2*(layer_cnt_tmp+1)]+[1], strides=[1]+pool_sizes[2*layer_cnt_tmp:2*(layer_cnt_tmp+1)]+[1], padding=padding_type)
            else:
                processed_skip_connect_input = None

            if layer_cnt == 0:
                if layer_types[layer_cnt] > -0.5:
                    if control_flag[0]:
                        dfcnn_layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_ELLA_cnn_deconv_tensordot_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                    elif control_flag[1]:
                        dfcnn_layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_ELLA_cnn_deconv_tensordot_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                    elif control_flag[2]:
                        dfcnn_layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_ELLA_cnn_deconv_tensordot_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                    elif control_flag[3]:
                        dfcnn_layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_ELLA_cnn_deconv_tensordot_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=None, TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                if layer_types[layer_cnt] < 0.5:
                    conv_layer_tmp, para_tmp = new_cnn_layer(layer_input=net_input, k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=cnn_params[2*layer_cnt], bias=cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable)
            else:
                if layer_types[layer_cnt] > -0.5:
                    if control_flag[0]:
                        dfcnn_layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_ELLA_cnn_deconv_tensordot_layer(cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], [ch_sizes[layer_cnt]*2, ch_sizes[layer_cnt+1]] if mixture_type.lower()=='concat' and layer_types[layer_cnt-1]>-0.5 and layer_types[layer_cnt-1]<0.5 else ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                    elif control_flag[1]:
                        dfcnn_layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_ELLA_cnn_deconv_tensordot_layer(cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], [ch_sizes[layer_cnt]*2, ch_sizes[layer_cnt+1]] if mixture_type.lower()=='concat' and layer_types[layer_cnt-1]>-0.5 and layer_types[layer_cnt-1]<0.5 else ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                    elif control_flag[2]:
                        dfcnn_layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_ELLA_cnn_deconv_tensordot_layer(cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], [ch_sizes[layer_cnt]*2, ch_sizes[layer_cnt+1]] if mixture_type.lower()=='concat' and layer_types[layer_cnt-1]>-0.5 and layer_types[layer_cnt-1]<0.5 else ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                    elif control_flag[3]:
                        dfcnn_layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_ELLA_cnn_deconv_tensordot_layer(cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], [ch_sizes[layer_cnt]*2, ch_sizes[layer_cnt+1]] if mixture_type.lower()=='concat' and layer_types[layer_cnt-1]>-0.5 and layer_types[layer_cnt-1]<0.5 else ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=None, TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                if layer_types[layer_cnt] < 0.5:
                    conv_layer_tmp, para_tmp = new_cnn_layer(layer_input=cnn_model[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+([ch_sizes[layer_cnt]*2, ch_sizes[layer_cnt+1]] if mixture_type.lower()=='concat' and layer_types[layer_cnt-1]>-0.5 and layer_types[layer_cnt-1]<0.5 else ch_sizes[layer_cnt:layer_cnt+2]), stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=cnn_params[2*layer_cnt], bias=cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable)
            if layer_types[layer_cnt] > 0.5:
                layer_tmp = dfcnn_layer_tmp
            elif layer_types[layer_cnt] < -0.5:
                layer_tmp = conv_layer_tmp
            else:
                weight_on_dfcnn = tf.nn.sigmoid(cnn_sharing[layer_cnt])
                if mixture_type.lower() == 'sum':
                    layer_tmp = weight_on_dfcnn*dfcnn_layer_tmp + (1.0-weight_on_dfcnn)*conv_layer_tmp
                elif mixture_type.lower() == 'concat':
                    layer_tmp = tf.concat([weight_on_dfcnn*dfcnn_layer_tmp, (1.0-weight_on_dfcnn)*conv_layer_tmp], axis=-1)
                else:
                    raise NotImplementedError("Invalid type of combining the output of Conv and DF-Conv layer!")

            cnn_model.append(layer_tmp)
            layers_for_skip.append(layer_tmp)
            cnn_KB_to_return = cnn_KB_to_return + KB_para_tmp
            cnn_TS_to_return = cnn_TS_to_return + TS_para_tmp
            cnn_params_to_return = cnn_params_to_return + para_tmp
            cnn_gen_params = cnn_gen_params + cnn_gen_para_tmp
            cnn_highway_params_to_return = cnn_highway_params_to_return + highway_para_tmp
            if layer_cnt == skip_connect_out:
                next_skip_connect = None

        #### flattening output
        output_dim = [int(cnn_model[-1].shape[1]*cnn_model[-1].shape[2]*cnn_model[-1].shape[3])]
        cnn_model.append(tf.reshape(cnn_model[-1], [-1, output_dim[0]]))

        #### add dropout layer
        if dropout:
            cnn_model.append(tf.nn.dropout(cnn_model[-1], dropout_prob))

    ## add fc layers
    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, tensorboard_name_scope='fc_net', trainable=trainable)

    #return (cnn_model+fc_model, cnn_KB_params, cnn_TS_params, cnn_gen_params, cnn_params_to_return, cnn_highway_params_to_return, fc_params)
    return (cnn_model+fc_model, cnn_KB_to_return, cnn_TS_to_return, cnn_gen_params, cnn_params_to_return, cnn_highway_params_to_return, fc_params, cnn_sharing)
