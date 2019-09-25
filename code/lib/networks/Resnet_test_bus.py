# coded by syshin

import tensorflow as tf
from networks.network import Network
from fast_rcnn.config import cfg

#define

n_classes = 3
_feat_stride = [16,]
anchor_scales = [8, 16, 32]

class Resnet_test_bus(Network):
    def __init__(self, net_depth, dataset, norm_type, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.layers = dict({'data':self.data, 'im_info':self.im_info})
        self.net_depth = net_depth
        self.dataset = dataset
        self.norm_type = norm_type
        self.trainable = trainable

        self.resnet_def = {
        18: ([2, 2, 2, 2], self.basicblock),
        34: ([3, 4, 6, 3], self.basicblock),
        50: ([3, 4, 6, 3], self.bottleneck),
        101: ([3, 4, 23, 3], self.bottleneck)
        }
        
        defs, block_func = self.resnet_def[net_depth]        
        self.defs = defs
        self.block_func = block_func
        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, norm_type=self.norm_type, is_training=False, name='conv0')
             .max_pool(3, 3, 2, 2, padding='VALID', name='group0/pool')
             .layer_group(self.block_func, 64, self.defs[0], 1, 1, norm_type=self.norm_type, is_training=False, first=True, name='group0')
             .layer_group(self.block_func, 128, self.defs[1], 2, 2, norm_type=self.norm_type, is_training=False, name='group1')
             .layer_group(self.block_func, 256, self.defs[2], 2, 2, norm_type=self.norm_type, is_training=False, name='group2')
             .normalization(norm_type=self.norm_type, is_training=False, name='group2/norm')
             .relu(name='group2/relu'))
 
        #========= RPN ============
        (self.feed('group2/relu')
             .conv(3, 3, (256 if self.block_func==self.basicblock else 256*4), 1, 1, norm_type=self.norm_type, is_training=False, name='rpn_conv/3x3')
             .conv(1, 1, len(anchor_scales)*3*2, 1, 1, norm_type=None, use_relu=False, is_training=False, padding='VALID', name='rpn_cls_score'))

        (self.feed('rpn_conv/3x3')
             .conv(1, 1, len(anchor_scales)*3*4, 1, 1, norm_type=None, use_relu=False, is_training=False, padding='VALID', name='rpn_bbox_pred'))

        (self.feed('rpn_cls_score')
             .reshape_layer(2, name='rpn_cls_score_reshape')
             .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
             .reshape_layer(len(anchor_scales)*3*2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
             .proposal_layer(_feat_stride, anchor_scales, False, False, name='rois'))

        #========= RCNN ============
        (self.feed('group2/relu', 'rois')
             .roi_pool(7, 7, 1.0/16, name='roi_pool')
             .layer_group(self.block_func, 512, self.defs[3], 2, 2, norm_type=self.norm_type, is_training=False, name='group3')
             .normalization(norm_type=self.norm_type, is_training=False, name='group3/norm')
             .relu(name='group3/relu')
             .GlobalAvgPooling(name='gap')
             .fc(n_classes, use_relu=False, name='cls_score')
             .softmax(name='cls_prob')) 

        (self.feed('gap')
             .fc(n_classes*4, use_relu=False, name='bbox_pred'))