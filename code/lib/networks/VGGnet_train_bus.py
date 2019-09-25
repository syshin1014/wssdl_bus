# coded by syshin
# modified from the original code 'VGGnet_train.py'

import tensorflow as tf
from networks.network import Network
from fast_rcnn.config import cfg

#define

n_classes = 3
_feat_stride = [16,]
anchor_scales = [8, 16, 32]

class VGGnet_train_bus(Network):
    def __init__(self, dataset, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 4])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, cfg.TRAIN.MAX_GT_PER_IMAGE, 5])
        self.num_gt_boxes = tf.placeholder(tf.int32, shape=[None]) # added for handling an arbitrary batch size  
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        self.is_ws = tf.placeholder(tf.bool)
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes, 'num_gt_boxes':self.num_gt_boxes, \
                            'keep_prob':self.keep_prob, 'is_training':self.is_training, 'is_ws':self.is_ws})
        self.dataset = dataset    
        self.trainable = trainable
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
             .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
             .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
             .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3'))
        #========= RPN ============
        (self.feed('conv5_3')
             .conv(3,3,512,1,1,name='rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*3*2 ,1 , 1, padding='VALID', use_relu=False, name='rpn_cls_score'))

        (self.feed('rpn_cls_score','gt_boxes','num_gt_boxes','im_info','data')
             .anchor_target_layer_joint(_feat_stride, anchor_scales, self.dataset, self.layers['is_training'], name = 'rpn-data' ))

        # Loss of rpn_cls & rpn_boxes

        (self.feed('rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*3*4, 1, 1, padding='VALID', use_relu=False, name='rpn_bbox_pred'))

        #========= RoI Proposal ============
        (self.feed('rpn_cls_score')
             .reshape_layer(2,name = 'rpn_cls_score_reshape')
             .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
             .reshape_layer(len(anchor_scales)*3*2,name = 'rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
             .proposal_layer(_feat_stride, anchor_scales, self.layers['is_training'], self.layers['is_ws'], name = 'rpn_rois'))  

        # beyond RPN
        (self.feed('rpn_rois','gt_boxes','num_gt_boxes')
             .proposal_target_layer_joint(n_classes, self.layers['is_training'], name = 'roi-data'))

        #========= RCNN ============
        (self.feed('conv5_3', 'roi-data')
             .roi_pool(7, 7, 1.0/16, name='pool_5')
             .fc(512, name='fc6')
             .dropout(self.layers['keep_prob'], name='drop6')
             .fc(512, name='fc7')
             .dropout(self.layers['keep_prob'], name='drop7')
             .fc(n_classes, use_relu=False, name='cls_score')
             .softmax(name='cls_prob')) 

        (self.feed('drop7')
             .fc(n_classes*4, use_relu=False, name='bbox_pred'))
         # beyond RPN