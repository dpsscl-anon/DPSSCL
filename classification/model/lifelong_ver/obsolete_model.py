import tensorflow as tf
import numpy as np

from utils.utils import get_list_of_valid_tensors, get_value_of_valid_tensors, savemat_wrapper, savemat_wrapper_nested_list, count_trainable_var2, cat_sigmoid_entropy, new_ELLA_KB_param
from utils.utils_df_nn import new_ELLA_flexible_cnn_deconv_tensordot_fc_net
from utils.utils_obsolete import new_Hybrid_DFCNN_net_auto_sharing
from classification.model.lifelong_ver.lifelong_model_frame import Lifelong_Model_Frame

_tf_ver = tf.__version__.split('.')
_up_to_date_tf = int(_tf_ver[0]) > 1 or (int(_tf_ver[0])==1 and int(_tf_ver[1]) >= 14)
if _up_to_date_tf:
    _tf_tensor = tf.is_tensor
else:
    _tf_tensor = tf.contrib.framework.is_tensor

################################################################
####      Hybrid DF-CNN with automatic sharing via mask     ####
################################################################
class LL_hybrid_DFCNN_automatic_sharing(Lifelong_Model_Frame):
    def __init__(self, model_hyperpara, train_hyperpara):
        super().__init__(model_hyperpara, train_hyperpara)
        #self.conv_sharing = model_hyperpara['conv_sharing']
        self.conv_sharing = []
        self.conv_sharing_bias = model_hyperpara['conv_sharing_bias']
        self.dfcnn_KB_size = model_hyperpara['cnn_KB_sizes']
        self.dfcnn_TS_size = model_hyperpara['cnn_TS_sizes']
        self.dfcnn_stride_size = model_hyperpara['cnn_deconv_stride_sizes']
        self.dfcnn_KB_reg_scale = model_hyperpara['regularization_scale'][1]
        self.dfcnn_TS_reg_scale = model_hyperpara['regularization_scale'][3]
        self.selection_loss_scale = model_hyperpara['selection_loss_scale']
        self.selection_var_scale = model_hyperpara['selection_var_scale']

    def _build_task_model(self, net_input, output_size, task_cnt, params=None, trainable=False):
        if params is None:
            params_KB, params_TS, params_conv, params_fc, params_gen_conv, params_sharing = None, None, None, None, None, None
        else:
            params_KB, params_TS, params_conv, params_fc, params_gen_conv, params_sharing = params['KB'], params['TS'], params['Conv'], params['FC'], params['ConvGen'], params['Sharing']

        if params_KB is not None:
            assert (len(params_KB) == self.num_conv_layers), "Given trained parameters of DF KB doesn't match to the hyper-parameters!"
        if params_TS is not None:
            assert (len(params_TS) == 4*self.num_conv_layers), "Given trained parameters of DF TS doesn't match to the hyper-parameters!"
        if params_conv is not None:
            assert (len(params_conv) == 2*self.num_conv_layers), "Given trained parameters of conv doesn't match to the hyper-parameters!"
        else:
            params_conv = [None for _ in range(2*self.num_conv_layers)]
        if params_fc is not None:
            assert (len(params_fc) == 2*self.num_fc_layers), "Given trained parameters of fc doesn't match to the hyper-parameters!"
        else:
            params_fc = [None for _ in range(2*self.num_fc_layers)]

        if task_cnt == self.current_task:
            if self.task_is_new:
                params_conv_for_init = params_conv
            else:
                assert (len(params_conv)==len(params_gen_conv)), "Number of conv params and generated conv params don't match!"
                params_conv_for_init = [a if a is not None else b for (a, b) in zip(params_conv, params_gen_conv)]
            task_net, dfcnn_KB_param_tmp, dfcnn_TS_param_tmp, gen_conv_param_tmp, conv_params, _, fc_params, sharing_params = new_Hybrid_DFCNN_net_auto_sharing(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], self.dfcnn_KB_size, self.dfcnn_TS_size, self.dfcnn_stride_size, sharing_params=params_sharing, cnn_activation_fn=self.hidden_act, cnn_para_activation_fn=None, cnn_KB_params=params_KB, cnn_TS_params=params_TS, cnn_params=params_conv_for_init, fc_activation_fn=self.hidden_act, fc_params=params_fc, KB_reg_type=self.KB_l2_reg, TS_reg_type=self.TS_l2_reg, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, sharing_var_scale=self.selection_var_scale, task_index=task_cnt, skip_connections=list(self.skip_connect), trainable=trainable, layer_types=self.conv_sharing_bias)
            self.dfcnn_sharing_selection_param = sharing_params
        else:
            conv_sharing = list(params_sharing > 0.0)
            task_net, dfcnn_KB_param_tmp, dfcnn_TS_param_tmp, gen_conv_param_tmp, conv_params, _, fc_params = new_ELLA_flexible_cnn_deconv_tensordot_fc_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], conv_sharing, self.dfcnn_KB_size, self.dfcnn_TS_size, self.dfcnn_stride_size, cnn_activation_fn=self.hidden_act, cnn_para_activation_fn=None, cnn_KB_params=params_KB, cnn_TS_params=params_TS, cnn_params=params_conv, fc_activation_fn=self.hidden_act, fc_params=params_fc, KB_reg_type=self.KB_l2_reg, TS_reg_type=self.TS_l2_reg, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, task_index=task_cnt, skip_connections=list(self.skip_connect), trainable=trainable)
        self.dfcnn_TS_params.append(dfcnn_TS_param_tmp)
        self.dfcnn_gen_conv_params.append(gen_conv_param_tmp)
        return task_net, conv_params, fc_params

    def _build_whole_model(self):
        for task_cnt, (num_classes, x_b) in enumerate(zip(self.output_sizes, self.x_batch)):
            if (task_cnt==self.current_task) and (self.task_is_new):
                param_to_reuse = {'KB': self.dfcnn_KB_params, 'TS': None, 'Conv': None, 'FC': None, 'ConvGen': None, 'Sharing': np.zeros(self.num_conv_layers, dtype=np.float32)}
            else:
                param_to_reuse = {'KB': self.dfcnn_KB_params, 'TS': self.np_params[task_cnt]['TS'], 'Conv': self.np_params[task_cnt]['Conv'], 'FC': self.np_params[task_cnt]['FC'], 'ConvGen': self.np_params[task_cnt]['ConvGen'], 'Sharing': self.conv_sharing[task_cnt]}
            task_net, conv_params, fc_params = self._build_task_model(x_b, num_classes, task_cnt, params=param_to_reuse, trainable=(task_cnt==self.current_task))

            if task_cnt == 0:
                self.dfcnn_KB_params_size = count_trainable_var2(self.dfcnn_KB_params)

            self.task_models.append(task_net)
            self.conv_params.append(conv_params)
            self.fc_params.append(fc_params)
            self.params.append(self._collect_trainable_variables())
            self.num_trainable_var += count_trainable_var2(self.params[-1]) if task_cnt < 1 else count_trainable_var2(self.params[-1]) - self.dfcnn_KB_params_size

        self.dfcnn_KB_trainable_param = get_list_of_valid_tensors(self.dfcnn_KB_params)
        self.dfcnn_TS_trainable_param = get_list_of_valid_tensors(self.dfcnn_TS_params[self.current_task])
        self.conv_trainable_param = get_list_of_valid_tensors(self.conv_params[self.current_task])
        self.fc_trainable_param = get_list_of_valid_tensors(self.fc_params[self.current_task])
        self.selection_param = get_list_of_valid_tensors(self.dfcnn_sharing_selection_param)

    def add_new_task(self, output_dim, curr_task_index, single_input_placeholder=False):
        self.dfcnn_KB_params, self.dfcnn_TS_params, self.dfcnn_gen_conv_params = None, [], []
        self.KB_l2_reg = tf.contrib.layers.l2_regularizer(scale=self.dfcnn_KB_reg_scale)
        self.TS_l2_reg = tf.contrib.layers.l2_regularizer(scale=self.dfcnn_TS_reg_scale)
        KB_init_val = self.np_params[0]['KB'] if hasattr(self, 'np_params') else [None for _ in range(self.num_conv_layers)]
        self.dfcnn_KB_params = [new_ELLA_KB_param([1, self.dfcnn_KB_size[2*layer_cnt], self.dfcnn_KB_size[2*layer_cnt], self.dfcnn_KB_size[2*layer_cnt+1]], layer_cnt, 0, self.KB_l2_reg, KB_init_val[layer_cnt], True) if self.conv_sharing_bias[layer_cnt] > -0.5 else None for layer_cnt in range(self.num_conv_layers)]
        super().add_new_task(output_dim, curr_task_index, single_input_placeholder=single_input_placeholder)

    def define_opt(self):
        with tf.name_scope('Optimization'):
            reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            KB_reg_term2 = tf.contrib.layers.apply_regularization(self.KB_l2_reg, reg_var)
            TS_reg_term2 = tf.contrib.layers.apply_regularization(self.TS_l2_reg, reg_var)

            KB_grads = tf.gradients(self.loss[self.current_task] + KB_reg_term2, self.dfcnn_KB_trainable_param)
            KB_grads_vars = [(grad, param) for grad, param in zip(KB_grads, self.dfcnn_KB_trainable_param)]

            TS_grads = tf.gradients(self.loss[self.current_task] + TS_reg_term2, self.dfcnn_TS_trainable_param)
            TS_grads_vars = [(grad, param) for grad, param in zip(TS_grads, self.dfcnn_TS_trainable_param)]

            conv_grads = tf.gradients(self.loss[self.current_task], self.conv_trainable_param)
            conv_grads_vars = [(grad, param) for grad, param in zip(conv_grads, self.conv_trainable_param)]

            fc_grads = tf.gradients(self.loss[self.current_task], self.fc_trainable_param)
            fc_grads_vars = [(grad, param) for grad, param in zip(fc_grads, self.fc_trainable_param)]

            selection_loss = self.selection_loss_scale*cat_sigmoid_entropy(tf.stack(self.selection_param))
            selection_grads = tf.gradients(self.loss[self.current_task] + selection_loss, self.selection_param)
            selection_grads_vars = [(grad, param) for grad, param in zip(selection_grads, self.selection_param)]

            self.grads = list(KB_grads) + list(TS_grads) + list(conv_grads) + list(fc_grads) + list(selection_grads)
            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay))
            grads_vars = KB_grads_vars + TS_grads_vars + conv_grads_vars + fc_grads_vars + selection_grads_vars
            self.update = trainer.apply_gradients(grads_vars)

    def get_params_val(self, sess):
        shared_layers_val = sess.run(self.dfcnn_sharing_selection_param)
        KB_param_val = get_value_of_valid_tensors(sess, self.dfcnn_KB_params)
        TS_param_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.dfcnn_TS_params]
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.dfcnn_gen_conv_params]
        cnn_param_val = [get_value_of_valid_tensors(sess, cnn_param) for cnn_param in self.conv_params]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_params]

        parameters_val = {}
        parameters_val['conv_sharing_weights'] = savemat_wrapper(shared_layers_val)
        parameters_val['conv_KB'] = savemat_wrapper(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['conv_trainable_weights'] = savemat_wrapper_nested_list(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val

    def convert_tfVar_to_npVar(self, sess):
        if not (self.num_tasks == 1 and self.task_is_new):
            orig_KB = list(self.np_params[0]['KB'])    ## copy of KB before training current task
        else:
            orig_KB = [None for _ in range(self.num_conv_layers)]

        def list_param_converter(list_of_params):
            converted_params = []
            for p in list_of_params:
                if type(p) == np.ndarray:
                    converted_params.append(p)
                elif _tf_tensor(p):
                    converted_params.append(sess.run(p))
                else:
                    converted_params.append(p)  ## append 'None' param
            return converted_params

        def double_list_param_converter(list_of_params):
            converted_params = []
            for task_params in list_of_params:
                converted_params.append(list_param_converter(task_params))
            return converted_params

        def post_process(layers_to_share, original_KB, updated_KB, updated_TS, updated_conv, gen_conv):
            for layer_cnt, (sharing_flag) in enumerate(layers_to_share):
                if sharing_flag:
                    ### Sharing this layer -> use new KB, TS and generated conv (no action needed), and make conv param None
                    updated_conv[self.current_task][2*layer_cnt], updated_conv[self.current_task][2*layer_cnt+1] = None, None
                else:
                    ### Not sharing this layer -> roll back KB, make TS and generated conv None, and keep conv param (no action needed)
                    updated_KB[layer_cnt] = original_KB[layer_cnt]
                    updated_TS[self.current_task][4*layer_cnt], updated_TS[self.current_task][4*layer_cnt+1] = None, None
                    updated_TS[self.current_task][4*layer_cnt+2], updated_TS[self.current_task][4*layer_cnt+3] = None, None
                    gen_conv[self.current_task][2*layer_cnt], gen_conv[self.current_task][2*layer_cnt+1] = None, None
            return updated_KB, updated_TS, updated_conv, gen_conv

        self.np_params = []
        shared_layers_val = np.array(sess.run(self.dfcnn_sharing_selection_param))
        if len(self.conv_sharing) < self.num_tasks:
            self.conv_sharing.append(shared_layers_val)
        else:
            self.conv_sharing[self.current_task] = shared_layers_val
        np_KB = list_param_converter(self.dfcnn_KB_params)
        np_TS = double_list_param_converter(self.dfcnn_TS_params)
        np_conv = double_list_param_converter(self.conv_params)
        np_fc = double_list_param_converter(self.fc_params)
        np_gen = double_list_param_converter(self.dfcnn_gen_conv_params)

        np_KB, np_TS, np_conv, np_gen = post_process(shared_layers_val>0.0, orig_KB, np_KB, np_TS, np_conv, np_gen)

        for t, g, c, f in zip(np_TS, np_gen, np_conv, np_fc):
            self.np_params.append({'KB': np_KB, 'TS': t, 'ConvGen': g, 'Conv': c, 'FC': f} if len(self.np_params)< 1 else {'TS': t, 'ConvGen': g, 'Conv': c, 'FC': f})

    def _collect_trainable_variables(self):
        return_list = []
        for p in self.dfcnn_KB_params:
            if p is not None:
                return_list.append(p)
        for p in self.dfcnn_TS_params[-1]:
            if p is not None:
                return_list.append(p)
        for p in self.conv_params[-1]:
            if p is not None:
                return_list.append(p)
        for p in self.fc_params[-1]:
            if p is not None:
                return_list.append(p)
        return return_list


########################################################
####      Hybrid DF-CNN with automatic sharing      ####
########################################################
class LL_hybrid_DFCNN_automatic_sharing_ver2(Lifelong_Model_Frame):
    def __init__(self, model_hyperpara, train_hyperpara):
        super().__init__(model_hyperpara, train_hyperpara)
        #self.conv_sharing = model_hyperpara['conv_sharing']
        self.conv_sharing = []
        self.conv_sharing_bias = model_hyperpara['conv_sharing_bias']
        self.dfcnn_KB_size = model_hyperpara['cnn_KB_sizes']
        self.dfcnn_TS_size = model_hyperpara['cnn_TS_sizes']
        self.dfcnn_stride_size = model_hyperpara['cnn_deconv_stride_sizes']
        self.dfcnn_KB_reg_scale = model_hyperpara['regularization_scale'][1]
        self.dfcnn_TS_reg_scale = model_hyperpara['regularization_scale'][3]
        self.selection_loss_scale = model_hyperpara['selection_loss_scale']
        self.selection_var_scale = model_hyperpara['selection_var_scale']
        self.train_phase1 = False

    def _build_task_model(self, net_input, output_size, task_cnt, params=None, trainable=False):
        if params is None:
            params_KB, params_TS, params_conv, params_fc, params_gen_conv, params_sharing = None, None, None, None, None, None
        else:
            params_KB, params_TS, params_conv, params_fc, params_gen_conv, params_sharing = params['KB'], params['TS'], params['Conv'], params['FC'], params['ConvGen'], params['Sharing']

        if params_KB is not None:
            assert (len(params_KB) == self.num_conv_layers), "Given trained parameters of DF KB doesn't match to the hyper-parameters!"
        if params_TS is not None:
            assert (len(params_TS) == 4*self.num_conv_layers), "Given trained parameters of DF TS doesn't match to the hyper-parameters!"
        if params_conv is not None:
            assert (len(params_conv) == 2*self.num_conv_layers), "Given trained parameters of conv doesn't match to the hyper-parameters!"
        else:
            params_conv = [None for _ in range(2*self.num_conv_layers)]
        if params_fc is not None:
            assert (len(params_fc) == 2*self.num_fc_layers), "Given trained parameters of fc doesn't match to the hyper-parameters!"
        else:
            params_fc = [None for _ in range(2*self.num_fc_layers)]

        if (task_cnt == self.current_task) and self.train_phase1:
            ## auto sharing
            task_net, dfcnn_KB_param_tmp, dfcnn_TS_param_tmp, gen_conv_param_tmp, conv_params, _, fc_params, sharing_params = new_Hybrid_DFCNN_net_auto_sharing(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], self.dfcnn_KB_size, self.dfcnn_TS_size, self.dfcnn_stride_size, sharing_params=params_sharing, cnn_activation_fn=self.hidden_act, cnn_para_activation_fn=None, cnn_KB_params=params_KB, cnn_TS_params=params_TS, cnn_params=params_conv, fc_activation_fn=self.hidden_act, fc_params=params_fc, KB_reg_type=self.KB_l2_reg, TS_reg_type=self.TS_l2_reg, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, sharing_var_scale=self.selection_var_scale, task_index=task_cnt, skip_connections=list(self.skip_connect), trainable=trainable, layer_types=self.conv_sharing_bias)
            self.dfcnn_sharing_selection_param = sharing_params
        else:
            ## hybrid DF-CNN
            conv_sharing = list(params_sharing > 0.0)
            task_net, dfcnn_KB_param_tmp, dfcnn_TS_param_tmp, gen_conv_param_tmp, conv_params, _, fc_params = new_ELLA_flexible_cnn_deconv_tensordot_fc_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], conv_sharing, self.dfcnn_KB_size, self.dfcnn_TS_size, self.dfcnn_stride_size, cnn_activation_fn=self.hidden_act, cnn_para_activation_fn=None, cnn_KB_params=params_KB, cnn_TS_params=params_TS, cnn_params=params_conv, fc_activation_fn=self.hidden_act, fc_params=params_fc, KB_reg_type=self.KB_l2_reg, TS_reg_type=self.TS_l2_reg, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, task_index=task_cnt, skip_connections=list(self.skip_connect), trainable=trainable)
        self.dfcnn_TS_params.append(dfcnn_TS_param_tmp)
        self.dfcnn_gen_conv_params.append(gen_conv_param_tmp)
        return task_net, conv_params, fc_params

    def _build_whole_model(self):
        for task_cnt, (num_classes, x_b) in enumerate(zip(self.output_sizes, self.x_batch)):
            if (task_cnt==self.current_task) and (self.task_is_new) and (self.train_phase1):
                param_to_reuse = {'KB': self.dfcnn_KB_params, 'TS': None, 'Conv': None, 'FC': None, 'ConvGen': None, 'Sharing': np.zeros(self.num_conv_layers, dtype=np.float32)}
            elif (task_cnt==self.current_task) and (self.task_is_new):
                param_to_reuse = {'KB': self.dfcnn_KB_params, 'TS': None, 'Conv': None, 'FC': None, 'ConvGen': None, 'Sharing': self.conv_sharing[task_cnt]}
            else:
                param_to_reuse = {'KB': self.dfcnn_KB_params, 'TS': self.np_params[task_cnt]['TS'], 'Conv': self.np_params[task_cnt]['Conv'], 'FC': self.np_params[task_cnt]['FC'], 'ConvGen': self.np_params[task_cnt]['ConvGen'], 'Sharing': self.conv_sharing[task_cnt]}
            task_net, conv_params, fc_params = self._build_task_model(x_b, num_classes, task_cnt, params=param_to_reuse, trainable=(task_cnt==self.current_task))

            if task_cnt == 0:
                self.dfcnn_KB_params_size = count_trainable_var2(self.dfcnn_KB_params)

            self.task_models.append(task_net)
            self.conv_params.append(conv_params)
            self.fc_params.append(fc_params)
            self.params.append(self._collect_trainable_variables())
            self.num_trainable_var += count_trainable_var2(self.params[-1]) if task_cnt < 1 else count_trainable_var2(self.params[-1]) - self.dfcnn_KB_params_size

        self.dfcnn_KB_trainable_param = get_list_of_valid_tensors(self.dfcnn_KB_params)
        self.dfcnn_TS_trainable_param = get_list_of_valid_tensors(self.dfcnn_TS_params[self.current_task])
        self.conv_trainable_param = get_list_of_valid_tensors(self.conv_params[self.current_task])
        self.fc_trainable_param = get_list_of_valid_tensors(self.fc_params[self.current_task])
        if self.train_phase1:
            self.selection_param = get_list_of_valid_tensors(self.dfcnn_sharing_selection_param)

    def add_new_task(self, output_dim, curr_task_index, single_input_placeholder=False):
        self.dfcnn_KB_params, self.dfcnn_TS_params, self.dfcnn_gen_conv_params = None, [], []
        self.KB_l2_reg = tf.contrib.layers.l2_regularizer(scale=self.dfcnn_KB_reg_scale)
        self.TS_l2_reg = tf.contrib.layers.l2_regularizer(scale=self.dfcnn_TS_reg_scale)
        KB_init_val = self.np_params[0]['KB'] if hasattr(self, 'np_params') else [None for _ in range(self.num_conv_layers)]
        self.dfcnn_KB_params = [new_ELLA_KB_param([1, self.dfcnn_KB_size[2*layer_cnt], self.dfcnn_KB_size[2*layer_cnt], self.dfcnn_KB_size[2*layer_cnt+1]], layer_cnt, 0, self.KB_l2_reg, KB_init_val[layer_cnt], True) if self.conv_sharing_bias[layer_cnt] > -0.5 else None for layer_cnt in range(self.num_conv_layers)]
        super().add_new_task(output_dim, curr_task_index, single_input_placeholder=single_input_placeholder)

    def define_opt(self):
        with tf.name_scope('Optimization'):
            reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            KB_reg_term2 = tf.contrib.layers.apply_regularization(self.KB_l2_reg, reg_var)
            TS_reg_term2 = tf.contrib.layers.apply_regularization(self.TS_l2_reg, reg_var)

            KB_grads = tf.gradients(self.loss[self.current_task] + KB_reg_term2, self.dfcnn_KB_trainable_param)
            KB_grads_vars = [(grad, param) for grad, param in zip(KB_grads, self.dfcnn_KB_trainable_param)]

            TS_grads = tf.gradients(self.loss[self.current_task] + TS_reg_term2, self.dfcnn_TS_trainable_param)
            TS_grads_vars = [(grad, param) for grad, param in zip(TS_grads, self.dfcnn_TS_trainable_param)]

            conv_grads = tf.gradients(self.loss[self.current_task], self.conv_trainable_param)
            conv_grads_vars = [(grad, param) for grad, param in zip(conv_grads, self.conv_trainable_param)]

            fc_grads = tf.gradients(self.loss[self.current_task], self.fc_trainable_param)
            fc_grads_vars = [(grad, param) for grad, param in zip(fc_grads, self.fc_trainable_param)]

            if self.train_phase1:
                selection_loss = self.selection_loss_scale*cat_sigmoid_entropy(tf.stack(self.selection_param))
                selection_grads = tf.gradients(self.loss[self.current_task] + selection_loss, self.selection_param)
                selection_grads_vars = [(2.5*grad, param) for grad, param in zip(selection_grads, self.selection_param)]

            if self.train_phase1:
                self.grads = list(KB_grads) + list(TS_grads) + list(conv_grads) + list(fc_grads) + list(selection_grads)
                grads_vars = KB_grads_vars + TS_grads_vars + conv_grads_vars + fc_grads_vars + selection_grads_vars
            else:
                self.grads = list(KB_grads) + list(TS_grads) + list(conv_grads) + list(fc_grads)
                grads_vars = KB_grads_vars + TS_grads_vars + conv_grads_vars + fc_grads_vars
            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay))
            self.update = trainer.apply_gradients(grads_vars)

    def get_params_val(self, sess):
        shared_layers_val = sess.run(self.dfcnn_sharing_selection_param) if self.train_phase1 else np.array(self.conv_sharing)
        KB_param_val = get_value_of_valid_tensors(sess, self.dfcnn_KB_params)
        TS_param_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.dfcnn_TS_params]
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.dfcnn_gen_conv_params]
        cnn_param_val = [get_value_of_valid_tensors(sess, cnn_param) for cnn_param in self.conv_params]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_params]

        parameters_val = {}
        parameters_val['conv_sharing_weights'] = savemat_wrapper(shared_layers_val)
        parameters_val['conv_KB'] = savemat_wrapper(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['conv_trainable_weights'] = savemat_wrapper_nested_list(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val

    def convert_tfVar_to_npVar(self, sess):
        if not (self.num_tasks == 1 and self.task_is_new):
            orig_KB = list(self.np_params[0]['KB'])    ## copy of KB before training current task
        else:
            orig_KB = [None for _ in range(self.num_conv_layers)]

        def list_param_converter(list_of_params):
            converted_params = []
            for p in list_of_params:
                if type(p) == np.ndarray:
                    converted_params.append(p)
                elif _tf_tensor(p):
                    converted_params.append(sess.run(p))
                else:
                    converted_params.append(p)  ## append 'None' param
            return converted_params

        def double_list_param_converter(list_of_params):
            converted_params = []
            for task_params in list_of_params:
                converted_params.append(list_param_converter(task_params))
            return converted_params

        self.np_params = []
        np_KB = list_param_converter(self.dfcnn_KB_params)
        np_TS = double_list_param_converter(self.dfcnn_TS_params)
        np_conv = double_list_param_converter(self.conv_params)
        np_fc = double_list_param_converter(self.fc_params)
        np_gen = double_list_param_converter(self.dfcnn_gen_conv_params)

        for t, g, c, f in zip(np_TS, np_gen, np_conv, np_fc):
            self.np_params.append({'KB': np_KB, 'TS': t, 'ConvGen': g, 'Conv': c, 'FC': f} if len(self.np_params)< 1 else {'TS': t, 'ConvGen': g, 'Conv': c, 'FC': f})

    def _collect_trainable_variables(self):
        return_list = []
        for p in self.dfcnn_KB_params:
            if p is not None:
                return_list.append(p)
        for p in self.dfcnn_TS_params[-1]:
            if p is not None:
                return_list.append(p)
        for p in self.conv_params[-1]:
            if p is not None:
                return_list.append(p)
        for p in self.fc_params[-1]:
            if p is not None:
                return_list.append(p)
        return return_list

    def learning_phase1(self, sess, curr_task_index, train_data, output_dim, param_folder_path, debug_mode, max_epoch):
        from random import shuffle
        from scipy.io import savemat

        batch_size, num_data = self.batch_size, train_data[0].shape[0]
        data_indices = list(range(num_data))

        self.train_phase1 = True
        self.add_new_task(output_dim, curr_task_index)
        task_model_index = self.find_task_model(curr_task_index)

        sess.run(tf.global_variables_initializer())
        if debug_mode:
            para_file_name = param_folder_path + '/phase1_init_param_task%d.mat'%(curr_task_index)
            curr_param = self.get_params_val(sess)
            savemat(para_file_name, {'parameter': curr_param})

        learning_step = 0
        while learning_step < max_epoch:
            learning_step = learning_step+1
            shuffle(data_indices)

            for batch_cnt in range(num_data//batch_size):
                batch_train_x = train_data[0][data_indices[batch_cnt*batch_size:(batch_cnt+1)*batch_size], :]
                batch_train_y = train_data[1][data_indices[batch_cnt*batch_size:(batch_cnt+1)*batch_size]]

                sess.run(self.update, feed_dict={self.model_input[task_model_index]: batch_train_x, self.true_output[task_model_index]: batch_train_y, self.epoch: learning_step-1, self.dropout_prob: 0.5})

        if debug_mode:
            para_file_name = param_folder_path + '/phase1_param_task%d.mat'%(curr_task_index)
            curr_param = self.get_params_val(sess)
            savemat(para_file_name, {'parameter': curr_param})

        ## Post-process
        self.train_phase1 = False
        shared_layers = np.array(sess.run(self.dfcnn_sharing_selection_param))
        self.conv_sharing.append(shared_layers)

        self.num_tasks -= 1
        _ = self.output_sizes.pop(-1)
        _ = self.task_indices.pop(-1)


########################################################
####      Hybrid DF-CNN with automatic sharing      ####
########################################################
class LL_hybrid_DFCNN_automatic_sharing_ver3(Lifelong_Model_Frame):
    def __init__(self, model_hyperpara, train_hyperpara):
        super().__init__(model_hyperpara, train_hyperpara)
        #self.conv_sharing = model_hyperpara['conv_sharing']
        self.conv_sharing = []
        self.conv_sharing_bias = model_hyperpara['conv_sharing_bias']
        self.dfcnn_KB_size = model_hyperpara['cnn_KB_sizes']
        self.dfcnn_TS_size = model_hyperpara['cnn_TS_sizes']
        self.dfcnn_stride_size = model_hyperpara['cnn_deconv_stride_sizes']
        self.dfcnn_KB_reg_scale = model_hyperpara['regularization_scale'][1]
        self.dfcnn_TS_reg_scale = model_hyperpara['regularization_scale'][3]
        self.selection_loss_scale = model_hyperpara['selection_loss_scale']
        self.selection_var_scale = model_hyperpara['selection_var_scale']

    def _build_task_model(self, net_input, output_size, task_cnt, params=None, trainable=False):
        if params is None:
            params_KB, params_TS, params_conv, params_fc, params_gen_conv, params_sharing = None, None, None, None, None, None
        else:
            params_KB, params_TS, params_conv, params_fc, params_gen_conv, params_sharing = params['KB'], params['TS'], params['Conv'], params['FC'], params['ConvGen'], params['Sharing']

        if params_KB is not None:
            assert (len(params_KB) == self.num_conv_layers), "Given trained parameters of DF KB doesn't match to the hyper-parameters!"
        if params_TS is not None:
            assert (len(params_TS) == 4*self.num_conv_layers), "Given trained parameters of DF TS doesn't match to the hyper-parameters!"
        if params_conv is not None:
            assert (len(params_conv) == 2*self.num_conv_layers), "Given trained parameters of conv doesn't match to the hyper-parameters!"
        else:
            params_conv = [None for _ in range(2*self.num_conv_layers)]
        if params_fc is not None:
            assert (len(params_fc) == 2*self.num_fc_layers), "Given trained parameters of fc doesn't match to the hyper-parameters!"
        else:
            params_fc = [None for _ in range(2*self.num_fc_layers)]

        if (task_cnt == self.current_task) and self.task_is_new:
            task_net, dfcnn_KB_param_tmp, dfcnn_TS_param_tmp, gen_conv_param_tmp, conv_params, _, fc_params, sharing_params = new_Hybrid_DFCNN_net_auto_sharing(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], self.dfcnn_KB_size, self.dfcnn_TS_size, self.dfcnn_stride_size, sharing_params=params_sharing, cnn_activation_fn=self.hidden_act, cnn_para_activation_fn=None, cnn_KB_params=params_KB, cnn_TS_params=params_TS, cnn_params=params_conv, fc_activation_fn=self.hidden_act, fc_params=params_fc, KB_reg_type=self.KB_l2_reg, TS_reg_type=self.TS_l2_reg, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, sharing_var_scale=self.selection_var_scale, task_index=task_cnt, skip_connections=list(self.skip_connect), trainable=trainable, mixture_type='concat', layer_types=self.conv_sharing_bias)
            self.dfcnn_sharing_selection_param = sharing_params
        else:
            conv_sharing = list(params_sharing > 0.0)
            task_net, dfcnn_KB_param_tmp, dfcnn_TS_param_tmp, gen_conv_param_tmp, conv_params, _, fc_params = new_ELLA_flexible_cnn_deconv_tensordot_fc_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], conv_sharing, self.dfcnn_KB_size, self.dfcnn_TS_size, self.dfcnn_stride_size, cnn_activation_fn=self.hidden_act, cnn_para_activation_fn=None, cnn_KB_params=params_KB, cnn_TS_params=params_TS, cnn_params=params_conv, fc_activation_fn=self.hidden_act, fc_params=params_fc, KB_reg_type=self.KB_l2_reg, TS_reg_type=self.TS_l2_reg, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, task_index=task_cnt, skip_connections=list(self.skip_connect), trainable=trainable)
        self.dfcnn_TS_params.append(dfcnn_TS_param_tmp)
        self.dfcnn_gen_conv_params.append(gen_conv_param_tmp)
        return task_net, conv_params, fc_params

    def _build_whole_model(self):
        for task_cnt, (num_classes, x_b) in enumerate(zip(self.output_sizes, self.x_batch)):
            if (task_cnt==self.current_task) and (self.task_is_new):
                param_to_reuse = {'KB': self.dfcnn_KB_params, 'TS': None, 'Conv': None, 'FC': None, 'ConvGen': None, 'Sharing': np.zeros(self.num_conv_layers, dtype=np.float32)}
            else:
                param_to_reuse = {'KB': self.dfcnn_KB_params, 'TS': self.np_params[task_cnt]['TS'], 'Conv': self.np_params[task_cnt]['Conv'], 'FC': self.np_params[task_cnt]['FC'], 'ConvGen': self.np_params[task_cnt]['ConvGen'], 'Sharing': self.conv_sharing[task_cnt]}
            task_net, conv_params, fc_params = self._build_task_model(x_b, num_classes, task_cnt, params=param_to_reuse, trainable=(task_cnt==self.current_task))

            if task_cnt == 0:
                self.dfcnn_KB_params_size = count_trainable_var2(self.dfcnn_KB_params)

            self.task_models.append(task_net)
            self.conv_params.append(conv_params)
            self.fc_params.append(fc_params)
            self.params.append(self._collect_trainable_variables())
            self.num_trainable_var += count_trainable_var2(self.params[-1]) if task_cnt < 1 else count_trainable_var2(self.params[-1]) - self.dfcnn_KB_params_size

        self.dfcnn_KB_trainable_param = get_list_of_valid_tensors(self.dfcnn_KB_params)
        self.dfcnn_TS_trainable_param = get_list_of_valid_tensors(self.dfcnn_TS_params[self.current_task])
        self.conv_trainable_param = get_list_of_valid_tensors(self.conv_params[self.current_task])
        self.fc_trainable_param = get_list_of_valid_tensors(self.fc_params[self.current_task])
        self.selection_param = get_list_of_valid_tensors(self.dfcnn_sharing_selection_param)

    def add_new_task(self, output_dim, curr_task_index, single_input_placeholder=False):
        self.dfcnn_KB_params, self.dfcnn_TS_params, self.dfcnn_gen_conv_params = None, [], []
        self.KB_l2_reg = tf.contrib.layers.l2_regularizer(scale=self.dfcnn_KB_reg_scale)
        self.TS_l2_reg = tf.contrib.layers.l2_regularizer(scale=self.dfcnn_TS_reg_scale)
        KB_init_val = self.np_params[0]['KB'] if hasattr(self, 'np_params') else [None for _ in range(self.num_conv_layers)]
        self.dfcnn_KB_params = [new_ELLA_KB_param([1, self.dfcnn_KB_size[2*layer_cnt], self.dfcnn_KB_size[2*layer_cnt], self.dfcnn_KB_size[2*layer_cnt+1]], layer_cnt, 0, self.KB_l2_reg, KB_init_val[layer_cnt], True) if self.conv_sharing_bias[layer_cnt] > -0.5 else None for layer_cnt in range(self.num_conv_layers)]
        super().add_new_task(output_dim, curr_task_index, single_input_placeholder=single_input_placeholder)

    def define_opt(self):
        with tf.name_scope('Optimization'):
            reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            KB_reg_term2 = tf.contrib.layers.apply_regularization(self.KB_l2_reg, reg_var)
            TS_reg_term2 = tf.contrib.layers.apply_regularization(self.TS_l2_reg, reg_var)

            KB_grads = tf.gradients(self.loss[self.current_task] + KB_reg_term2, self.dfcnn_KB_trainable_param)
            KB_grads_vars = [(grad, param) for grad, param in zip(KB_grads, self.dfcnn_KB_trainable_param)]

            TS_grads = tf.gradients(self.loss[self.current_task] + TS_reg_term2, self.dfcnn_TS_trainable_param)
            TS_grads_vars = [(grad, param) for grad, param in zip(TS_grads, self.dfcnn_TS_trainable_param)]

            conv_grads = tf.gradients(self.loss[self.current_task], self.conv_trainable_param)
            conv_grads_vars = [(grad, param) for grad, param in zip(conv_grads, self.conv_trainable_param)]

            fc_grads = tf.gradients(self.loss[self.current_task], self.fc_trainable_param)
            fc_grads_vars = [(grad, param) for grad, param in zip(fc_grads, self.fc_trainable_param)]

            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay))
            if self.task_is_new:
                selection_loss = self.selection_loss_scale*cat_sigmoid_entropy(tf.stack(self.selection_param))
                selection_grads = tf.gradients(self.loss[self.current_task] + selection_loss, self.selection_param)
                selection_grads_vars = [(grad, param) for grad, param in zip(selection_grads, self.selection_param)]

                self.grads = list(KB_grads) + list(TS_grads) + list(conv_grads) + list(fc_grads) + list(selection_grads)
                grads_vars = KB_grads_vars + TS_grads_vars + conv_grads_vars + fc_grads_vars + selection_grads_vars
            else:
                self.grads = list(KB_grads) + list(TS_grads) + list(conv_grads) + list(fc_grads)
                grads_vars = KB_grads_vars + TS_grads_vars + conv_grads_vars + fc_grads_vars
            self.update = trainer.apply_gradients(grads_vars)

    def get_params_val(self, sess):
        #shared_layers_val = sess.run(self.dfcnn_sharing_selection_param)
        KB_param_val = get_value_of_valid_tensors(sess, self.dfcnn_KB_params)
        TS_param_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.dfcnn_TS_params]
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.dfcnn_gen_conv_params]
        cnn_param_val = [get_value_of_valid_tensors(sess, cnn_param) for cnn_param in self.conv_params]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_params]

        parameters_val = {}
        #parameters_val['conv_sharing_weights'] = savemat_wrapper(shared_layers_val)
        parameters_val['conv_sharing_weights'] = self.conv_sharing
        parameters_val['conv_KB'] = savemat_wrapper(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['conv_trainable_weights'] = savemat_wrapper_nested_list(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val

    def convert_tfVar_to_npVar(self, sess):
        if not (self.num_tasks == 1 and self.task_is_new):
            orig_KB = list(self.np_params[0]['KB'])    ## copy of KB before training current task
        else:
            orig_KB = [None for _ in range(self.num_conv_layers)]

        def list_param_converter(list_of_params):
            converted_params = []
            for p in list_of_params:
                if type(p) == np.ndarray:
                    converted_params.append(p)
                elif _tf_tensor(p):
                    converted_params.append(sess.run(p))
                else:
                    converted_params.append(p)  ## append 'None' param
            return converted_params

        def double_list_param_converter(list_of_params):
            converted_params = []
            for task_params in list_of_params:
                converted_params.append(list_param_converter(task_params))
            return converted_params

        def post_process(layers_to_share, original_KB, updated_KB, updated_TS, updated_conv, gen_conv, updated_fc):
            for layer_cnt, (sharing_flag) in enumerate(layers_to_share):
                if sharing_flag:
                    ### Sharing this layer -> use new KB, TS and generated conv (no action needed), and make conv param None
                    ### slice U of TS and generated conv based on sharing of lower layer
                    updated_conv[self.current_task][2*layer_cnt], updated_conv[self.current_task][2*layer_cnt+1] = None, None
                    if layer_cnt > 0:
                        num_ch = updated_TS[self.current_task][4*layer_cnt+2].shape[1]
                        sliced_weight = updated_TS[self.current_task][4*layer_cnt+2][:, 0:num_ch//2, :] if layers_to_share[layer_cnt-1] else updated_TS[self.current_task][4*layer_cnt+2][:, num_ch//2:, :]
                        updated_TS[self.current_task][4*layer_cnt+2] = sliced_weight
                        sliced_weight2 = gen_conv[self.current_task][2*layer_cnt][:, :, 0:num_ch//2, :] if layers_to_share[layer_cnt-1] else gen_conv[self.current_task][2*layer_cnt][:, :, num_ch//2:, :]
                        gen_conv[self.current_task][2*layer_cnt] = sliced_weight2
                else:
                    ### Not sharing this layer -> roll back KB, make TS and generated conv None, and keep conv param (no action needed)
                    ### slice conv param based on sharing of lower layer
                    updated_KB[layer_cnt] = original_KB[layer_cnt]
                    updated_TS[self.current_task][4*layer_cnt], updated_TS[self.current_task][4*layer_cnt+1] = None, None
                    updated_TS[self.current_task][4*layer_cnt+2], updated_TS[self.current_task][4*layer_cnt+3] = None, None
                    gen_conv[self.current_task][2*layer_cnt], gen_conv[self.current_task][2*layer_cnt+1] = None, None
                    if layer_cnt > 0:
                        num_ch = updated_conv[self.current_task][2*layer_cnt].shape[2]
                        sliced_weight = updated_conv[self.current_task][2*layer_cnt][:, :, 0:num_ch//2, :] if layers_to_share[layer_cnt-1] else updated_conv[self.current_task][2*layer_cnt][:, :, num_ch//2:, :]
                        updated_conv[self.current_task][2*layer_cnt] = sliced_weight

            num_ch, num_hidden = updated_TS[self.current_task][-2].shape[-1] if layers_to_share[-1] else updated_conv[self.current_task][-2].shape[-1], updated_fc[self.current_task][0].shape[1]
            W_tmp = np.array(updated_fc[self.current_task][0]).reshape([-1, 2*num_ch, num_hidden])
            sliced_weight = W_tmp[:, 0:num_ch, :] if layers_to_share[-1] else W_tmp[:, num_ch:, :]
            updated_fc[self.current_task][0] = sliced_weight.reshape([-1, num_hidden])
            return updated_KB, updated_TS, updated_conv, gen_conv, updated_fc

        self.np_params = []
        np_KB = list_param_converter(self.dfcnn_KB_params)
        np_TS = double_list_param_converter(self.dfcnn_TS_params)
        np_conv = double_list_param_converter(self.conv_params)
        np_fc = double_list_param_converter(self.fc_params)
        np_gen = double_list_param_converter(self.dfcnn_gen_conv_params)

        if self.task_is_new:
            shared_layers_val = np.array(sess.run(self.dfcnn_sharing_selection_param))
            self.conv_sharing.append(shared_layers_val)
            np_KB, np_TS, np_conv, np_gen, np_fc = post_process(shared_layers_val>0.0, orig_KB, np_KB, np_TS, np_conv, np_gen, np_fc)

        for t, g, c, f in zip(np_TS, np_gen, np_conv, np_fc):
            self.np_params.append({'KB': np_KB, 'TS': t, 'ConvGen': g, 'Conv': c, 'FC': f} if len(self.np_params)< 1 else {'TS': t, 'ConvGen': g, 'Conv': c, 'FC': f})

    def _collect_trainable_variables(self):
        return_list = []
        for p in self.dfcnn_KB_params:
            if p is not None:
                return_list.append(p)
        for p in self.dfcnn_TS_params[-1]:
            if p is not None:
                return_list.append(p)
        for p in self.conv_params[-1]:
            if p is not None:
                return_list.append(p)
        for p in self.fc_params[-1]:
            if p is not None:
                return_list.append(p)
        return return_list


########################################################
####      Hybrid DF-CNN with automatic sharing      ####
########################################################
class LL_hybrid_DFCNN_automatic_sharing_ver4(Lifelong_Model_Frame):
    def __init__(self, model_hyperpara, train_hyperpara):
        super().__init__(model_hyperpara, train_hyperpara)
        #self.conv_sharing = model_hyperpara['conv_sharing']
        self.conv_sharing = []
        self.conv_sharing_bias = model_hyperpara['conv_sharing_bias']
        self.dfcnn_KB_size = model_hyperpara['cnn_KB_sizes']
        self.dfcnn_TS_size = model_hyperpara['cnn_TS_sizes']
        self.dfcnn_stride_size = model_hyperpara['cnn_deconv_stride_sizes']
        self.dfcnn_KB_reg_scale = model_hyperpara['regularization_scale'][1]
        self.dfcnn_TS_reg_scale = model_hyperpara['regularization_scale'][3]
        self.selection_loss_scale = model_hyperpara['selection_loss_scale']
        self.selection_var_scale = model_hyperpara['selection_var_scale']
        self.train_phase1 = False
        self.phase1_max_epoch = model_hyperpara['phase1_max_epoch']

    def _build_task_model(self, net_input, output_size, task_cnt, params=None, trainable=False):
        if params is None:
            params_KB, params_TS, params_conv, params_fc, params_gen_conv, params_sharing = None, None, None, None, None, None
        else:
            params_KB, params_TS, params_conv, params_fc, params_gen_conv, params_sharing = params['KB'], params['TS'], params['Conv'], params['FC'], params['ConvGen'], params['Sharing']

        if params_KB is not None:
            assert (len(params_KB) == self.num_conv_layers), "Given trained parameters of DF KB doesn't match to the hyper-parameters!"
        if params_TS is not None:
            assert (len(params_TS) == 4*self.num_conv_layers), "Given trained parameters of DF TS doesn't match to the hyper-parameters!"
        if params_conv is not None:
            assert (len(params_conv) == 2*self.num_conv_layers), "Given trained parameters of conv doesn't match to the hyper-parameters!"
        else:
            params_conv = [None for _ in range(2*self.num_conv_layers)]
        if params_fc is not None:
            assert (len(params_fc) == 2*self.num_fc_layers), "Given trained parameters of fc doesn't match to the hyper-parameters!"
        else:
            params_fc = [None for _ in range(2*self.num_fc_layers)]

        if (task_cnt == self.current_task) and self.task_is_new and self.train_phase1:
            ## auto sharing
            task_net, dfcnn_KB_param_tmp, dfcnn_TS_param_tmp, gen_conv_param_tmp, conv_params, _, fc_params, sharing_params = new_Hybrid_DFCNN_net_auto_sharing(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], self.dfcnn_KB_size, self.dfcnn_TS_size, self.dfcnn_stride_size, sharing_params=params_sharing, cnn_activation_fn=self.hidden_act, cnn_para_activation_fn=None, cnn_KB_params=params_KB, cnn_TS_params=params_TS, cnn_params=params_conv, fc_activation_fn=self.hidden_act, fc_params=params_fc, KB_reg_type=self.KB_l2_reg, TS_reg_type=self.TS_l2_reg, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, sharing_var_scale=self.selection_var_scale, task_index=task_cnt, skip_connections=list(self.skip_connect), trainable=trainable, mixture_type='concat', layer_types=self.conv_sharing_bias)
            self.dfcnn_sharing_selection_param = sharing_params
        else:
            ## hybrid DF-CNN
            conv_sharing = list(params_sharing > 0.0)
            task_net, dfcnn_KB_param_tmp, dfcnn_TS_param_tmp, gen_conv_param_tmp, conv_params, _, fc_params = new_ELLA_flexible_cnn_deconv_tensordot_fc_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], conv_sharing, self.dfcnn_KB_size, self.dfcnn_TS_size, self.dfcnn_stride_size, cnn_activation_fn=self.hidden_act, cnn_para_activation_fn=None, cnn_KB_params=params_KB, cnn_TS_params=params_TS, cnn_params=params_conv, fc_activation_fn=self.hidden_act, fc_params=params_fc, KB_reg_type=self.KB_l2_reg, TS_reg_type=self.TS_l2_reg, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, task_index=task_cnt, skip_connections=list(self.skip_connect), trainable=trainable)
        self.dfcnn_TS_params.append(dfcnn_TS_param_tmp)
        self.dfcnn_gen_conv_params.append(gen_conv_param_tmp)
        return task_net, conv_params, fc_params

    def _build_whole_model(self):
        for task_cnt, (num_classes, x_b) in enumerate(zip(self.output_sizes, self.x_batch)):
            if (task_cnt==self.current_task) and (self.task_is_new) and (self.train_phase1):
                param_to_reuse = {'KB': self.dfcnn_KB_params, 'TS': None, 'Conv': None, 'FC': None, 'ConvGen': None, 'Sharing': np.zeros(self.num_conv_layers, dtype=np.float32)}
            elif (task_cnt==self.current_task) and (self.task_is_new):
                param_to_reuse = {'KB': self.dfcnn_KB_params, 'TS': None, 'Conv': None, 'FC': None, 'ConvGen': None, 'Sharing': self.conv_sharing[task_cnt]}
            else:
                param_to_reuse = {'KB': self.dfcnn_KB_params, 'TS': self.np_params[task_cnt]['TS'], 'Conv': self.np_params[task_cnt]['Conv'], 'FC': self.np_params[task_cnt]['FC'], 'ConvGen': self.np_params[task_cnt]['ConvGen'], 'Sharing': self.conv_sharing[task_cnt]}
            task_net, conv_params, fc_params = self._build_task_model(x_b, num_classes, task_cnt, params=param_to_reuse, trainable=(task_cnt==self.current_task))

            if task_cnt == 0:
                self.dfcnn_KB_params_size = count_trainable_var2(self.dfcnn_KB_params)

            self.task_models.append(task_net)
            self.conv_params.append(conv_params)
            self.fc_params.append(fc_params)
            self.params.append(self._collect_trainable_variables())
            self.num_trainable_var += count_trainable_var2(self.params[-1]) if task_cnt < 1 else count_trainable_var2(self.params[-1]) - self.dfcnn_KB_params_size

        self.dfcnn_KB_trainable_param = get_list_of_valid_tensors(self.dfcnn_KB_params)
        self.dfcnn_TS_trainable_param = get_list_of_valid_tensors(self.dfcnn_TS_params[self.current_task])
        self.conv_trainable_param = get_list_of_valid_tensors(self.conv_params[self.current_task])
        self.fc_trainable_param = get_list_of_valid_tensors(self.fc_params[self.current_task])
        if self.train_phase1:
            self.selection_param = get_list_of_valid_tensors(self.dfcnn_sharing_selection_param)

    def add_new_task(self, output_dim, curr_task_index, single_input_placeholder=False):
        self.dfcnn_KB_params, self.dfcnn_TS_params, self.dfcnn_gen_conv_params = None, [], []
        self.KB_l2_reg = tf.contrib.layers.l2_regularizer(scale=self.dfcnn_KB_reg_scale)
        self.TS_l2_reg = tf.contrib.layers.l2_regularizer(scale=self.dfcnn_TS_reg_scale)
        KB_init_val = self.np_params[0]['KB'] if hasattr(self, 'np_params') else [None for _ in range(self.num_conv_layers)]
        self.dfcnn_KB_params = [new_ELLA_KB_param([1, self.dfcnn_KB_size[2*layer_cnt], self.dfcnn_KB_size[2*layer_cnt], self.dfcnn_KB_size[2*layer_cnt+1]], layer_cnt, 0, self.KB_l2_reg, KB_init_val[layer_cnt], True) if self.conv_sharing_bias[layer_cnt] > -0.5 else None for layer_cnt in range(self.num_conv_layers)]
        super().add_new_task(output_dim, curr_task_index, single_input_placeholder=single_input_placeholder)

    def define_opt(self):
        with tf.name_scope('Optimization'):
            reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            KB_reg_term2 = tf.contrib.layers.apply_regularization(self.KB_l2_reg, reg_var)
            TS_reg_term2 = tf.contrib.layers.apply_regularization(self.TS_l2_reg, reg_var)

            KB_grads = tf.gradients(self.loss[self.current_task] + KB_reg_term2, self.dfcnn_KB_trainable_param)
            KB_grads_vars = [(grad, param) for grad, param in zip(KB_grads, self.dfcnn_KB_trainable_param)]

            TS_grads = tf.gradients(self.loss[self.current_task] + TS_reg_term2, self.dfcnn_TS_trainable_param)
            TS_grads_vars = [(grad, param) for grad, param in zip(TS_grads, self.dfcnn_TS_trainable_param)]

            conv_grads = tf.gradients(self.loss[self.current_task], self.conv_trainable_param)
            conv_grads_vars = [(grad, param) for grad, param in zip(conv_grads, self.conv_trainable_param)]

            fc_grads = tf.gradients(self.loss[self.current_task], self.fc_trainable_param)
            fc_grads_vars = [(grad, param) for grad, param in zip(fc_grads, self.fc_trainable_param)]

            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay))
            if self.train_phase1:
                selection_loss = self.selection_loss_scale*cat_sigmoid_entropy(tf.stack(self.selection_param))
                selection_grads = tf.gradients(self.loss[self.current_task] + selection_loss, self.selection_param)
                selection_grads_vars = [(grad, param) for grad, param in zip(selection_grads, self.selection_param)]
                #self.selection_loss = selection_loss

                self.grads = list(KB_grads) + list(TS_grads) + list(conv_grads) + list(fc_grads) + list(selection_grads)
                grads_vars = KB_grads_vars + TS_grads_vars + conv_grads_vars + fc_grads_vars + selection_grads_vars
            else:
                self.grads = list(KB_grads) + list(TS_grads) + list(conv_grads) + list(fc_grads)
                grads_vars = KB_grads_vars + TS_grads_vars + conv_grads_vars + fc_grads_vars
            self.update = trainer.apply_gradients(grads_vars)

    def get_params_val(self, sess):
        #shared_layers_val = sess.run(self.dfcnn_sharing_selection_param) if self.train_phase1 else np.array(self.conv_sharing)
        KB_param_val = get_value_of_valid_tensors(sess, self.dfcnn_KB_params)
        TS_param_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.dfcnn_TS_params]
        gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.dfcnn_gen_conv_params]
        cnn_param_val = [get_value_of_valid_tensors(sess, cnn_param) for cnn_param in self.conv_params]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_params]

        parameters_val = {}
        #parameters_val['conv_sharing_weights'] = savemat_wrapper(shared_layers_val)
        parameters_val['conv_sharing_weights'] = self.conv_sharing
        parameters_val['conv_KB'] = savemat_wrapper(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['conv_trainable_weights'] = savemat_wrapper_nested_list(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val

    def convert_tfVar_to_npVar(self, sess):
        if not (self.num_tasks == 1 and self.task_is_new):
            orig_KB = list(self.np_params[0]['KB'])    ## copy of KB before training current task
        else:
            orig_KB = [None for _ in range(self.num_conv_layers)]

        def list_param_converter(list_of_params):
            converted_params = []
            for p in list_of_params:
                if type(p) == np.ndarray:
                    converted_params.append(p)
                elif _tf_tensor(p):
                    converted_params.append(sess.run(p))
                else:
                    converted_params.append(p)  ## append 'None' param
            return converted_params

        def double_list_param_converter(list_of_params):
            converted_params = []
            for task_params in list_of_params:
                converted_params.append(list_param_converter(task_params))
            return converted_params

        self.np_params = []
        np_KB = list_param_converter(self.dfcnn_KB_params)
        np_TS = double_list_param_converter(self.dfcnn_TS_params)
        np_conv = double_list_param_converter(self.conv_params)
        np_fc = double_list_param_converter(self.fc_params)
        np_gen = double_list_param_converter(self.dfcnn_gen_conv_params)

        for t, g, c, f in zip(np_TS, np_gen, np_conv, np_fc):
            self.np_params.append({'KB': np_KB, 'TS': t, 'ConvGen': g, 'Conv': c, 'FC': f} if len(self.np_params)< 1 else {'TS': t, 'ConvGen': g, 'Conv': c, 'FC': f})

    def _collect_trainable_variables(self):
        return_list = []
        for p in self.dfcnn_KB_params:
            if p is not None:
                return_list.append(p)
        for p in self.dfcnn_TS_params[-1]:
            if p is not None:
                return_list.append(p)
        for p in self.conv_params[-1]:
            if p is not None:
                return_list.append(p)
        for p in self.fc_params[-1]:
            if p is not None:
                return_list.append(p)
        return return_list

    def learning_phase1(self, sess, curr_task_index, train_data, output_dim, param_folder_path, debug_mode):
        from random import shuffle
        from scipy.io import savemat

        batch_size, num_data = self.batch_size, train_data[0].shape[0]
        data_indices = list(range(num_data))

        self.train_phase1 = True
        self.add_new_task(output_dim, curr_task_index)
        task_model_index = self.find_task_model(curr_task_index)

        sess.run(tf.global_variables_initializer())
        if debug_mode:
            para_file_name = param_folder_path + '/phase1_init_param_task%d.mat'%(curr_task_index)
            curr_param = self.get_params_val(sess)
            savemat(para_file_name, {'parameter': curr_param})

        learning_step = 0
        while learning_step < self.phase1_max_epoch:
            learning_step = learning_step+1
            shuffle(data_indices)

            for batch_cnt in range(num_data//batch_size):
                batch_train_x = train_data[0][data_indices[batch_cnt*batch_size:(batch_cnt+1)*batch_size], :]
                batch_train_y = train_data[1][data_indices[batch_cnt*batch_size:(batch_cnt+1)*batch_size]]

                sess.run(self.update, feed_dict={self.model_input[task_model_index]: batch_train_x, self.true_output[task_model_index]: batch_train_y, self.epoch: learning_step-1, self.dropout_prob: 0.5})
                #_, l1, l2 = sess.run([self.update, self.loss[task_model_index], self.selection_loss], feed_dict={self.model_input[task_model_index]: batch_train_x, self.true_output[task_model_index]: batch_train_y, self.epoch: learning_step-1, self.dropout_prob: 0.5})

        if debug_mode:
            para_file_name = param_folder_path + '/phase1_param_task%d.mat'%(curr_task_index)
            curr_param = self.get_params_val(sess)
            savemat(para_file_name, {'parameter': curr_param})

        ## Post-process
        self.train_phase1 = False
        shared_layers = np.array(sess.run(self.dfcnn_sharing_selection_param))
        self.conv_sharing.append(shared_layers)

        self.num_tasks -= 1
        _ = self.output_sizes.pop(-1)
        _ = self.task_indices.pop(-1)
