# coded by syshin

import tensorflow as tf
from networks.network import Network
from fast_rcnn.config import cfg

#define

n_classes = 3
_feat_stride = [16,]
anchor_scales = [8, 16, 32]

class Resnet_train_bus(Network):
    def __init__(self, net_depth, dataset, norm_type, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 4]) # modified for adding image-level labels (diagnostic label in this case)
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, cfg.TRAIN.MAX_GT_PER_IMAGE, 5]) # modified for handling an arbitrary batch size
        self.num_gt_boxes = tf.placeholder(tf.int32, shape=[None]) # added for handling an arbitrary batch size         
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool) # added by syshin
        self.is_ws = tf.placeholder(tf.bool) # added by syshin
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes, 'num_gt_boxes':self.num_gt_boxes, \
                            'keep_prob':self.keep_prob, 'is_training':self.is_training, 'is_ws':self.is_ws})
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

        # beyond RPN
        # create ops and placeholders for bbox normalization process
        with tf.variable_scope('bbox_pred', reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

            self.bbox_weights_assign = weights.assign(self.bbox_weights)
            self.bbox_bias_assign = biases.assign(self.bbox_biases)
        # beyond RPN

    def setup(self):
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, norm_type=self.norm_type, is_training=self.layers['is_training'], name='conv0')
             .max_pool(3, 3, 2, 2, padding='VALID', name='group0/pool')
             .layer_group(self.block_func, 64, self.defs[0], 1, 1, norm_type=self.norm_type, is_training=self.layers['is_training'], first=True, name='group0')
             .layer_group(self.block_func, 128, self.defs[1], 2, 2, norm_type=self.norm_type, is_training=self.layers['is_training'], name='group1')
             .layer_group(self.block_func, 256, self.defs[2], 2, 2, norm_type=self.norm_type, is_training=self.layers['is_training'], name='group2')
             .normalization(norm_type=self.norm_type, is_training=self.layers['is_training'], name='group2/norm')
             .relu(name='group2/relu'))
        
        #========= RPN ============
        (self.feed('group2/relu')
             .conv(3, 3, (256 if self.block_func==self.basicblock else 256*4), 1, 1, norm_type=self.norm_type, is_training=self.layers['is_training'], name='rpn_conv/3x3')
             .conv(1, 1, len(anchor_scales)*3*2, 1, 1, norm_type=None, use_relu=False, is_training=self.layers['is_training'], padding='VALID', name='rpn_cls_score'))

        (self.feed('rpn_cls_score','gt_boxes','num_gt_boxes','im_info','data')
             .anchor_target_layer_joint(_feat_stride, anchor_scales, self.dataset, self.layers['is_training'], name='rpn-data')) # different to the iterative one

        (self.feed('rpn_conv/3x3')
             .conv(1, 1, len(anchor_scales)*3*4, 1, 1, norm_type=None, use_relu=False, is_training=self.layers['is_training'], padding='VALID', name='rpn_bbox_pred'))

        (self.feed('rpn_cls_score')
             .reshape_layer(2, name='rpn_cls_score_reshape')
             .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
             .reshape_layer(len(anchor_scales)*3*2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
             .proposal_layer(_feat_stride, anchor_scales, self.layers['is_training'], self.layers['is_ws'], name='rpn_rois'))  

        # beyond RPN
        (self.feed('rpn_rois','gt_boxes','num_gt_boxes')
             .proposal_target_layer_joint(n_classes, self.layers['is_training'], name='roi-data')) # different to the iterative one

        #========= RCNN ============
        (self.feed('group2/relu', 'roi-data')
             .roi_pool(7, 7, 1.0/16, name='roi_pool')
             .layer_group(self.block_func, 512, self.defs[3], 2, 2, norm_type=self.norm_type, is_training=self.layers['is_training'], name='group3')
             .normalization(norm_type=self.norm_type, is_training=self.layers['is_training'], name='group3/norm')
             .relu(name='group3/relu')
             .GlobalAvgPooling(name='gap')
             .fc(n_classes, use_relu=False, name='cls_score')
             .softmax(name='cls_prob')) 

        (self.feed('gap')
             .fc(n_classes*4, use_relu=False, name='bbox_pred'))
        # beyond RPN