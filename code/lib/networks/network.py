# modified by syshin

import numpy as np
import tensorflow as tf
import roi_pooling_layer.roi_pooling_op as roi_pool_op
import roi_pooling_layer.roi_pooling_op_grad
from rpn_msr.proposal_layer_tf_bus import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer_tf_bus import anchor_target_layer as anchor_target_layer_py
from rpn_msr.anchor_target_layer_tf_bus import anchor_target_layer_ws as anchor_target_layer_ws_py
from rpn_msr.anchor_target_layer_tf_bus import anchor_target_layer_joint as anchor_target_layer_joint_py
from rpn_msr.proposal_target_layer_tf_bus import proposal_target_layer as proposal_target_layer_py
from rpn_msr.proposal_target_layer_tf_bus import proposal_target_layer_joint as proposal_target_layer_joint_py
from tensorflow.contrib.framework import add_model_variable
from tensorflow.python.training import moving_averages
from fast_rcnn.config import cfg


DEFAULT_PADDING = 'SAME'


def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path).item()
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print "assign pretrain model "+subkey+ " to "+key
                    except ValueError:
                        print "ignore "+key+"/"+subkey
                        #print "ignore "+key
                        if not ignore_missing:

                            raise
    
    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, norm_type=None, use_relu=True, is_training=True, padding=DEFAULT_PADDING, group=1, trainable=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            #init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable)
            #biases = self.make_var('biases', [c_o], init_biases, trainable)

            if group==1:
                out = convolve(input, kernel)
            else:
                input_groups = tf.split(axis=3, num_or_size_splits=group, value=input)
                kernel_groups = tf.split(axis=3, num_or_size_splits=group, value=kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                out = tf.concat(axis=3, values=output_groups)
           
            if norm_type=='BN':
                out = tf.layers.batch_normalization(out, training=is_training, renorm=cfg.TRAIN.USE_BRN)
            elif norm_type=='GN':
                #out = tf.contrib.layers.group_norm(out, groups=min(cfg.TRAIN.GN_MIN_NUM_G, c_o/cfg.TRAIN.GN_MIN_CHS_PER_G))
                out = self.group_norm(out, num_group=min(cfg.TRAIN.GN_MIN_NUM_G, c_o/cfg.TRAIN.GN_MIN_CHS_PER_G))
            else:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                out = tf.nn.bias_add(out, biases)
     
            if use_relu:
                out = tf.nn.relu(out)
        
        return out
    
    # for internal use only        
    def conv_int(self, input, k_h, k_w, c_o, s_h, s_w, name, norm_type=None, use_relu=True, is_training=True, padding=DEFAULT_PADDING, group=1, trainable=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            #init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable)
            #biases = self.make_var('biases', [c_o], init_biases, trainable)

            if group==1:
                out = convolve(input, kernel)
            else:
                input_groups = tf.split(axis=3, num_or_size_splits=group, value=input)
                kernel_groups = tf.split(axis=3, num_or_size_splits=group, value=kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                out = tf.concat(axis=3, values=output_groups)
           
            if norm_type=='BN':
                out = tf.layers.batch_normalization(out, training=is_training, renorm=cfg.TRAIN.USE_BRN)
            elif norm_type=='GN':
                #out = tf.contrib.layers.group_norm(out, groups=min(cfg.TRAIN.GN_MIN_NUM_G, c_o/cfg.TRAIN.GN_MIN_CHS_PER_G))
                out = self.group_norm(out, num_group=min(cfg.TRAIN.GN_MIN_NUM_G, c_o/cfg.TRAIN.GN_MIN_CHS_PER_G))
            else:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                out = tf.nn.bias_add(out, biases)
     
            if use_relu:
                out = tf.nn.relu(out)
        
        return out

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def roi_pool(self, input, pooled_height, pooled_width, spatial_scale, name):
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]

        print input
        return roi_pool_op.roi_pool(input[0], input[1],
                                    pooled_height,
                                    pooled_width,
                                    spatial_scale,
                                    name=name)[0]

    @layer
    def proposal_layer(self, input, _feat_stride, anchor_scales, is_training, is_ws, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        return tf.reshape(tf.py_func(proposal_layer_py,[input[0],input[1],input[2], is_training, is_ws, _feat_stride, anchor_scales], [tf.float32]),[-1,5],name =name)

    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, dataset, is_ws, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:

            rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = tf.cond(is_ws, lambda: tf.py_func(anchor_target_layer_ws_py,[input[0],input[1],input[2],input[3],input[4], _feat_stride, anchor_scales],[tf.float32,tf.float32,tf.float32,tf.float32]), \
                                                                                                          lambda: tf.py_func(anchor_target_layer_py,[input[0],input[1],input[2],input[3],input[4], _feat_stride, anchor_scales, dataset],[tf.float32,tf.float32,tf.float32,tf.float32]))
            
            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels')
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets')
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights')
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights')

            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    @layer
    def anchor_target_layer_joint(self, input, _feat_stride, anchor_scales, dataset, is_training, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:

            rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = tf.py_func(anchor_target_layer_joint_py,[input[0],input[1],input[2],input[3],input[4], is_training, _feat_stride, anchor_scales, dataset],[tf.float32,tf.float32,tf.float32,tf.float32])
            
            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels')
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets')
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights')
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights')

            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    @layer
    def proposal_target_layer(self, input, classes, is_training, is_ws, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:

            rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights = tf.py_func(proposal_target_layer_py,[input[0],input[1],input[2],classes,is_training,is_ws],[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])

            rois = tf.reshape(rois,[-1,5] , name = 'rois') 
            labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'labels')
            bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'bbox_targets')
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name = 'bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name = 'bbox_outside_weights')
           
            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    @layer
    def proposal_target_layer_joint(self, input, classes, is_training, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:

            rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights = tf.py_func(proposal_target_layer_joint_py,[input[0],input[1],input[2],classes,is_training],[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])

            rois = tf.reshape(rois,[-1,5] , name = 'rois') 
            labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'labels')
            bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'bbox_targets')
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name = 'bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name = 'bbox_outside_weights')

            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    @layer
    def reshape_layer(self, input, d,name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
                    int(d),tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),input_shape[2]]),[0,2,3,1],name=name)
        else:
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
                    int(d),tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),input_shape[2]]),[0,2,3,1],name=name)

    @layer
    def feature_extrapolating(self, input, scales_base, num_scale_base, num_per_octave, name):
        return feature_extrapolating_op.feature_extrapolating(input,
                              scales_base,
                              num_scale_base,
                              num_per_octave,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def fc(self, input, num_out, name, norm_type=None, use_relu=True, is_training=True, trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                #init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                #init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable)
            #biases = self.make_var('biases', [num_out], init_biases, trainable)
            
            out = tf.matmul(feed_in, weights)
            if norm_type=='BN':
                out = tf.layers.batch_normalization(out, training=is_training, renorm=cfg.TRAIN.USE_BRN)
            elif norm_type=='GN':
                #out = tf.contrib.layers.group_norm(out, groups=min(cfg.TRAIN.GN_MIN_NUM_G, num_out/cfg.TRAIN.GN_MIN_CHS_PER_G))
                out = self.group_norm(out, num_group=min(cfg.TRAIN.GN_MIN_NUM_G, num_out/cfg.TRAIN.GN_MIN_CHS_PER_G))
            else:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', [num_out], init_biases, trainable)
                out = out + biases
                
            if use_relu:
                out = tf.nn.relu(out) 

        return out

    # for internal use only
    def fc_int(self, input, num_out, name, norm_type=None, use_relu=True, is_training=True, trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                #init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                #init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable)
            #biases = self.make_var('biases', [num_out], init_biases, trainable)
            
            out = tf.matmul(feed_in, weights)
            if norm_type=='BN':
                out = tf.layers.batch_normalization(out, training=is_training, renorm=cfg.TRAIN.USE_BRN)
            elif norm_type=='GN':
                #out = tf.contrib.layers.group_norm(out, groups=min(cfg.TRAIN.GN_MIN_NUM_G, num_out/cfg.TRAIN.GN_MIN_CHS_PER_G))
                out = self.group_norm(out, num_group=min(cfg.TRAIN.GN_MIN_NUM_G, num_out/cfg.TRAIN.GN_MIN_CHS_PER_G))
            else:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', [num_out], init_biases, trainable)
                out = out + biases
                
            if use_relu:
                out = tf.nn.relu(out) 

        return out

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        else:
            return tf.nn.softmax(input,name=name)
            
    def softmax_int(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        else:
            return tf.nn.softmax(input,name=name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    # functions below are added for ResNet
    def shortcut(self, input, c_i, c_o, s_h, s_w, norm_type=None, is_training=True, trainable=True):
        if c_i != c_o:
            return self.conv_int(input, 1, 1, c_o, s_h, s_w, name='convshortcut', norm_type=norm_type, use_relu=False, is_training=is_training, trainable=trainable)                
        else:
            return input
    
    def basicblock(self, input, c_o, s_h, s_w, preact, norm_type=None, is_training=True, trainable=True):
        c_i = input.get_shape().as_list()[3]
        if preact == 'both_preact':
            input = self.normalization_int(input, name='preact_prev', norm_type=norm_type, is_training=is_training)
            input = tf.nn.relu(input, 'preact')           
            input_ori = input
        elif preact != 'no_preact':
            input_ori = input
            input = self.normalization_int(input, name='preact_prev', norm_type=norm_type, is_training=is_training)
            input = tf.nn.relu(input, 'preact')   
        else:
            input_ori = input
            
        input = self.conv_int(input, 3, 3, c_o, s_h, s_w, name='conv1', norm_type=norm_type, is_training=is_training, trainable=trainable)        
        input = self.conv_int(input, 3, 3, c_o, 1, 1, name='conv2', norm_type=norm_type, use_relu=False, is_training=is_training, trainable=trainable)
        return input + self.shortcut(input_ori, c_i, c_o, s_h, s_w, norm_type=norm_type, is_training=is_training, trainable=trainable)
    
    def bottleneck(self, input, c_o, s_h, s_w, preact, norm_type=None, is_training=True, trainable=True):
        c_i = input.get_shape().as_list()[3]
        if preact == 'both_preact':
            input = self.normalization_int(input, name='preact_prev', norm_type=norm_type, is_training=is_training)
            input = tf.nn.relu(input, 'preact')           
            input_ori = input
        elif preact != 'no_preact':
            input_ori = input
            input = self.normalization_int(input, name='preact_prev', norm_type=norm_type, is_training=is_training)
            input = tf.nn.relu(input, 'preact')  
        else:
            input_ori = input
        
        input = self.conv_int(input, 1, 1, c_o, 1, 1, name='conv1', norm_type=norm_type, is_training=is_training, trainable=trainable)
        input = self.conv_int(input, 3, 3, c_o, s_h, s_w, name='conv2', norm_type=norm_type, is_training=is_training, trainable=trainable)
        input = self.conv_int(input, 1, 1, c_o*4, 1, 1, name='conv3', norm_type=norm_type, use_relu=False, is_training=is_training, trainable=trainable)  
        return input + self.shortcut(input_ori, c_i, c_o*4, s_h, s_w, norm_type=norm_type, is_training=is_training, trainable=trainable) 
    
    @layer
    def layer_group(self, input, block_func, c_o, count, s_h, s_w, name, first=False, norm_type=None, is_training=True, trainable=True):
        with tf.variable_scope(name):
            with tf.variable_scope('block0'):
                input = block_func(input, c_o, s_h, s_w, ('no_preact' if first else 'both_preact'), \
                                   norm_type, is_training, trainable)
            for i in range(1, count):
                with tf.variable_scope('block{}'.format(i)):
                    input = block_func(input, c_o, 1, 1, 'default', norm_type, is_training, trainable)
            return input

    @layer
    def fc_group_parallel(self, input, num_outs, count, name, norm_type=None, use_relu=True, is_training=True, trainable=True):
        assert len(num_outs)==count
        out_list = []
        #with tf.variable_scope(name):
        for i in range(count):
            out_list.append(self.fc_int(input, num_outs[i], name+'{}'.format(i), norm_type=norm_type, use_relu=use_relu, is_training=is_training, trainable=trainable))
        return out_list
           
    @layer
    def softmax_group(self, input, name):
        out_list = []
        #with tf.variable_scope(name):
        for i in range(len(input)):
            out_list.append(self.softmax_int(input[i], name+'{}'.format(i)))
        return out_list
                
    @layer 
    def GlobalAvgPooling(self, input, name, data_format='NHWC'):
        """
        Global average pooling as in the paper `Network In Network
        <http://arxiv.org/abs/1312.4400>`_.
    
        Args:
            input (tf.Tensor): a NHWC tensor.
        Returns:
            tf.Tensor: a NC tensor named ``output``.
        """
        assert input.get_shape().ndims == 4
        assert data_format in ['NHWC', 'NCHW']
        axis = [1, 2] if data_format == 'NHWC' else [2, 3]
        return tf.reduce_mean(input, axis, name=name)
    
    @layer
    def normalization(self, input, name, norm_type=None, is_training=True):
        with tf.variable_scope(name) as scope:
            if norm_type=='BN':
                out = tf.layers.batch_normalization(input, training=is_training, renorm=cfg.TRAIN.USE_BRN)    
            elif norm_type=='GN':
                input_shape = input.get_shape().as_list()
                #out = tf.contrib.layers.group_norm(input, groups=min(cfg.TRAIN.GN_MIN_NUM_G, input_shape[-1]/cfg.TRAIN.GN_MIN_CHS_PER_G))
                out = self.group_norm(input, num_group=min(cfg.TRAIN.GN_MIN_NUM_G, input_shape[-1]/cfg.TRAIN.GN_MIN_CHS_PER_G))
            else:
                out = input
        return out
    
    def normalization_int(self, input, name, norm_type=None, is_training=True):
        with tf.variable_scope(name) as scope:
            if norm_type=='BN':
                out = tf.layers.batch_normalization(input, training=is_training, renorm=cfg.TRAIN.USE_BRN)    
            elif norm_type=='GN':
                input_shape = input.get_shape().as_list()
                #out = tf.contrib.layers.group_norm(input, groups=min(cfg.TRAIN.GN_MIN_NUM_G, input_shape[-1]/cfg.TRAIN.GN_MIN_CHS_PER_G))
                out = self.group_norm(input, num_group=min(cfg.TRAIN.GN_MIN_NUM_G, input_shape[-1]/cfg.TRAIN.GN_MIN_CHS_PER_G))
            else:
                out = input
        return out
    
    def group_norm(self, input, num_group=32, epsilon=1e-05):
        # We here assume the channel-last ordering (NHWC)
        
        num_ch = input.get_shape().as_list()[-1]
        num_group = min(num_group, num_ch)
        
        NHWCG = tf.concat([tf.slice(tf.shape(input),[0],[3]), tf.constant([num_ch//num_group, num_group])], axis=0)
        output = tf.reshape(input, NHWCG)
        
        mean, var = tf.nn.moments(output, [1, 2, 3], keep_dims=True)
        output = (output - mean) / tf.sqrt(var + epsilon)
        
        # gamma and beta
        gamma = tf.get_variable('gamma', [1, 1, 1, num_ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, num_ch], initializer=tf.constant_initializer(0.0))

        output = tf.reshape(output, tf.shape(input)) * gamma + beta
        
        return output