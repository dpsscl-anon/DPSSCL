import sys
import tensorflow as tf

if sys.version_info.major < 3:
    from utils_nn import new_cnn_fc_net
else:
    from utils.utils_nn import new_cnn_fc_net


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