# coded by syshin
# modified from the original code 'train.py'

"""Train a Fast R-CNN network."""

from fast_rcnn.config import cfg
#import gt_data_layer.roidb as gdl_roidb
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer_bus import RoIDataLayer
from roi_data_layer.layer_bus_joint import RoIDataLayerJoint
#from roi_data_layer.layer_bus_ws import WeaklySupervisedDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.python.client import timeline
import time
import mil.core as mil_core
import matplotlib.pyplot as plt
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.test_bus import vis_detections
import cPickle
import skimage.io
from utils.cython_nms import nms, nms_new
import pdb

#from AMSGrad import AMSGrad
from tensorflow.python.platform import tf_logging as logging


class ReduceLROnPlateau(object):
    """Mimicking the keras function with the same name.""" 
    def __init__(self, init_lr, monitor='val_loss', factor=0.5, patience=5, verbose=True, mode='auto', epsilon=1e-04, cooldown=0, min_lr=0):
        super(ReduceLROnPlateau, self).__init__()
        self.cur_lr = init_lr
        self.monitor = monitor
        if factor>=1.0:
            raise ValueError('ReduceLROnPlateau does not support a factor >= 1.0.')
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.epsilon = epsilon
        self.cooldown = cooldown
        self.min_lr = min_lr
        
        self.cooldown_counter = 0
        self.wait = 0
        self.best = 0
        self.monitor_op = None
        self._reset()
        
    def _reset(self):
        """Resets wait counter and cooldown counter."""
        if self.mode not in ['auto', 'min', 'max']:
            raise RuntimeError('Mode %s is unknown, fallback to auto mode.', self.mode)
        if (self.mode == 'min' or (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0
        
    def in_cooldown(self):
        return self.cooldown_counter > 0
    
    def on_val_end(self, logs):
        assert len(logs)!=0
        current = logs[-1]
    
        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            if self.wait >= self.patience:
                if self.cur_lr > self.min_lr:
                    new_lr = self.cur_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    self.cur_lr = new_lr
                    if self.verbose:
                        print('\nIter %06d: ReduceLROnPlateau reduced learning rate to %s.' % (cfg.TRAIN.TEST_ITERS*len(logs), new_lr))
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
            self.wait += 1
        
    def get_cur_lr(self):
        return self.cur_lr

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """
    
    def __init__(self, sess, network, imdb_train_s, roidb_train_s, imdb_train_ws, roidb_train_ws, imdb_test, roidb_test, output_dir, pretrained_model=None,
                 opt='adam', lr=5e-04, lr_scheduling='const', vis=False):
        """Initialize the SolverWrapper."""
        self.net = network
        temp = str(network.__class__)
        self.net_name = temp[temp.rfind('.')+1:-2]
        self.imdb_train_s = imdb_train_s
        self.roidb_train_s = roidb_train_s
        self.imdb_train_ws = imdb_train_ws
        self.roidb_train_ws = roidb_train_ws
        self.imdb_test = imdb_test
        self.roidb_test = roidb_test
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model
        self.opt = opt
        self.lr = lr
        self.lr_scheduling = lr_scheduling
        self.vis = vis
        
        # added for adaptive lr
        learning_rate = tf.placeholder(tf.float32, shape=[])
        self.learning_rate = learning_rate

        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb_train_s)
        print 'done'

        # For checkpoint
        self.saver = tf.train.Saver(max_to_keep=None)

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            """# scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0 * np.tile(self.bbox_stds, (weights_shape[0], 1))})
            sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 * self.bbox_stds + self.bbox_means})"""
            sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0})
            sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1})

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.net_name[:6] + '_fast_rcnn'+ infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        """filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')"""
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            with tf.variable_scope('bbox_pred', reuse=True):
                # restore net to original state
                sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0})
                sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1})

    def train_model_alter(self, sess, max_iters, s_start_iter, s_end_iter, ws_start_iter, ws_end_iter, max_per_image=300, thresh=0.05):
        """Network training loop."""

        global_step = tf.Variable(0, trainable=False)

        data_layer_train_s = get_data_layer(self.roidb_train_s, self.net_name, self.imdb_train_s.num_classes, is_training=True)
        data_layer_train_ws = get_data_layer(self.roidb_train_ws, self.net_name, self.imdb_train_ws.num_classes, is_training=True, is_ws=True)
        data_layer_test = get_data_layer(self.roidb_test, self.net_name, self.imdb_test.num_classes, is_training=False)

        # RPN
        # classification loss
        rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score_reshape'),[-1,2])
        rpn_label = tf.reshape(self.net.get_output('rpn-data')[0],[-1])
        # ignore_label(-1)
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label,-1))),[-1,2])
        rpn_label = tf.reshape(tf.gather(rpn_label,tf.where(tf.not_equal(rpn_label,-1))),[-1])
        # simple cross entropy loss #
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))
        # weighted cross entropy loss #
        """rpn_num_all = tf.cast(tf.size(rpn_label), dtype=tf.float32)
        rpn_num_fg = tf.count_nonzero(rpn_label, dtype=tf.float32)
        rpn_num_bg = rpn_num_all - rpn_num_fg
        rpn_class_weight = tf.multiply(2.,tf.cast(tf.concat(values=[tf.reshape(tf.divide(rpn_num_fg,rpn_num_all),(1,1)), \
                                        tf.reshape(tf.divide(rpn_num_bg,rpn_num_all),(1,1))], axis=1), dtype=tf.float32))
        rpn_weight_per_label = tf.transpose(tf.matmul(tf.one_hot(indices=rpn_label, depth=2),tf.transpose(rpn_class_weight))) #shape [1,N]
        rpn_cross_entropy = tf.reduce_mean(tf.multiply(rpn_weight_per_label,tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label)))"""

        # bounding box regression L1 loss
        rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
        rpn_bbox_targets = tf.transpose(self.net.get_output('rpn-data')[1],[0,2,3,1])
        rpn_bbox_inside_weights = tf.transpose(self.net.get_output('rpn-data')[2],[0,2,3,1])
        rpn_bbox_outside_weights = tf .transpose(self.net.get_output('rpn-data')[3],[0,2,3,1])
        smoothL1_sign = tf.cast(tf.less(tf.abs(tf.subtract(rpn_bbox_pred, rpn_bbox_targets)),1),tf.float32)
        rpn_loss_box = tf.multiply(tf.reduce_mean(tf.reduce_sum(tf.multiply(rpn_bbox_outside_weights,tf.add(
                       tf.multiply(tf.multiply(tf.pow(tf.multiply(rpn_bbox_inside_weights, tf.subtract(rpn_bbox_pred, rpn_bbox_targets))*3,2),0.5),smoothL1_sign),
                       tf.multiply(tf.subtract(tf.abs(tf.subtract(rpn_bbox_pred, rpn_bbox_targets)),0.5/9.0),tf.abs(smoothL1_sign-1)))), axis=[1,2])),10)

        # beyond RPN
        # R-CNN
        # classification loss
        cls_score = self.net.get_output('cls_score')
        label = tf.reshape(self.net.get_output('roi-data')[1],[-1])
        # simple cross entropy loss # 
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))
        # weighted cross entropy loss #
        #temp_weight_s = np.array([1, 1, (1-cfg.TRAIN.S_MAL_PCT)/cfg.TRAIN.S_MAL_PCT], dtype=np.float32)
        #temp_weight_s = [list(temp_weight_s/np.sum(temp_weight_s))]
        #class_weight_s = tf.constant(temp_weight_s)
        """hist = tf.add(1e-10,tf.divide(tf.cast(tf.histogram_fixed_width(tf.cast(label,tf.float64), tf.cast([0,self.imdb_train_s.num_classes],tf.float64), self.imdb_train_s.num_classes), \
                tf.float32), tf.cast(tf.size(label),tf.float32)))
        class_weight_s = tf.reshape(tf.cast(tf.divide(1./self.imdb_train_s.num_classes,hist),tf.float32),[1,self.imdb_train_s.num_classes])
        class_weight_s = tf.multiply(3.,tf.divide(class_weight_s,tf.reduce_sum(class_weight_s)))
        weight_per_label_s = tf.transpose(tf.matmul(tf.one_hot(label, depth=self.imdb_train_s.num_classes),tf.transpose(class_weight_s))) #shape [1,N]
        cross_entropy = tf.reduce_mean(tf.multiply(weight_per_label_s, tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label)))"""

        # bounding box regression L1 loss
        bbox_pred = self.net.get_output('bbox_pred')
        bbox_targets = self.net.get_output('roi-data')[2]
        bbox_inside_weights = self.net.get_output('roi-data')[3]
        bbox_outside_weights = self.net.get_output('roi-data')[4]
        loss_box = tf.reduce_mean(tf.reduce_sum(tf.multiply(bbox_outside_weights,tf.multiply(bbox_inside_weights, tf.abs(tf.subtract(bbox_pred, bbox_targets)))), axis=[1]))

        # MIL
        # classification loss
        batch_inds = tf.slice(self.net.get_output('roi-data')[0],[0,0],[-1,1])
        mil_label = tf.cast(tf.reshape(tf.slice(self.net.get_output('im_info'), [0,3], [-1,1]),[-1]),tf.int32) # image-level labels
        funcs = [mil_core.get_mass_max_logit, mil_core.get_mal_max_logit]
        """mil_cls_score, scale_factors = tf.cond(self.net.get_output('is_ws'), \
            lambda: mil_core.get_bag_logit(cls_score, batch_inds, self.imdb_train_s.num_classes, tf.reshape(mil_label,[-1,1]), cfg.TRAIN.WS_IMS_PER_BATCH, funcs), \
            lambda: mil_core.get_bag_logit(cls_score, batch_inds, self.imdb_train_s.num_classes, tf.reshape(mil_label,[-1,1]), cfg.TRAIN.IMS_PER_BATCH, funcs))"""
        mil_cls_score, _ = tf.cond(self.net.get_output('is_ws'), \
            lambda: mil_core.get_bag_logit(cls_score, batch_inds, self.imdb_train_s.num_classes, tf.reshape(mil_label,[-1,1]), cfg.TRAIN.WS_IMS_PER_BATCH, funcs), \
            lambda: mil_core.get_bag_logit(cls_score, batch_inds, self.imdb_train_s.num_classes, tf.reshape(mil_label,[-1,1]), cfg.TRAIN.IMS_PER_BATCH, funcs))
        scale_factors = tf.tile(tf.reshape(tf.subtract(1.,tf.train.exponential_decay(0.99, global_step, 2000, 0.9, staircase=True)),[-1]), tf.reshape(tf.size(mil_label),[-1])) 
        # simple cross entropy loss #
        #mil_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mil_cls_score, labels=mil_label))
        # weighted cross entropy loss #  
        temp_weight_ws = [[0, cfg.TRAIN.WS_MAL_PCT, 1-cfg.TRAIN.WS_MAL_PCT]]
        class_weight_ws = tf.constant(temp_weight_ws)
        #mal_weight_ws = tf.reshape(tf.divide(tf.reduce_sum(tf.cast(tf.equal(mil_label,1),tf.float32)),cfg.TRAIN.WS_IMS_PER_BATCH),(1,1))                                
        #class_weight_ws = tf.concat(values=[tf.zeros([1,1],tf.float32),tf.subtract(1.,mal_weight_ws),mal_weight_ws], axis=1)
        weight_per_label_ws = tf.transpose(tf.matmul(tf.one_hot(mil_label, depth=self.imdb_train_ws.num_classes),tf.transpose(class_weight_ws))) #shape [1,N]
        if cfg.TRAIN.WS_LOSS_USE_ADAPTIVE_SCALE_FACTOR:
            mil_cross_entropy = tf.reduce_mean(tf.multiply(scale_factors, tf.multiply(weight_per_label_ws, tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mil_cls_score, labels=mil_label))))
        else:
            mil_cross_entropy = tf.reduce_mean(tf.multiply(cfg.TRAIN.WS_LOSS_SCALE_FACTOR, tf.multiply(weight_per_label_ws, tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mil_cls_score, labels=mil_label))))
        # beyond RPN
        
        """loss = self.net.get_output('use_s_branch')*(cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box) + \
                self.net.get_output('use_ws_branch')*mil_cross_entropy"""
        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box        
        
        # added by syshin
        weights_only = filter( lambda x: x.name.endswith('weights:0'), tf.trainable_variables() )
        weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only])) * cfg.TRAIN.WEIGHT_DECAY
        loss += weight_decay
        # added by syshin
        
        #learning rate
        if self.lr_scheduling=='const':
            pass
        elif self.lr_scheduling=='pc':
            boundaries = [int(max_iters*0.75)]
            values = [self.lr,self.lr*0.1]
            pc_lr = tf.train.piecewise_constant(global_step, boundaries, values)            
        elif self.lr_scheduling=='rop':
            rop_handler = ReduceLROnPlateau(self.lr, monitor='val_loss', factor=0.5, patience=5, verbose=True, mode='min', epsilon=1e-03, cooldown=0, min_lr=0)
        else:
            NotImplemented
            
        # optimizer  
        if self.opt=='adam':
            train_op_s = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.1).minimize(loss) #s
            optimizer_ws = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.1) #ws
        elif self.opt=='amsgrad':
            train_op_s = AMSGrad(self.learning_rate, beta2=0.999, epsilon=0.1).minimize(loss) #s
            optimizer_ws = AMSGrad(self.learning_rate, beta2=0.999, epsilon=0.1) #ws
        elif self.opt=='sgd':
            train_op_s = tf.train.MomentumOptimizer(self.learning_rate, cfg.TRAIN.MOMENTUM, use_nesterov=True).minimize(loss, global_step=global_step) #s
            optimizer_ws = tf.train.MomentumOptimizer(self.learning_rate, cfg.TRAIN.MOMENTUM, use_nesterov=True) #ws
        else:
            NotImplemented

        grads_and_vars = optimizer_ws.compute_gradients(mil_cross_entropy)

        #grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
        train_op_ws = optimizer_ws.apply_gradients(grads_and_vars, global_step=global_step)

        summary_writer = tf.summary.FileWriter(self.output_dir, sess.graph) # added by syshin

        # initialize variables
        sess.run(tf.global_variables_initializer())        
        #sess.run(tf.initialize_all_variables())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)

        thresh = 0.05
        max_per_image=300
        last_snapshot_iter = -1
        timer = Timer()
        # added by syshin
        f_log = open(os.path.join(self.output_dir,'log.txt'), 'w')
        len_training_s = len(self.roidb_train_s) # length of the augmented tranining set
        len_training_ws = len(self.roidb_train_ws) # length of the augmented tranining set
        len_test = len(self.roidb_test)
        training_loss = np.zeros((6,), dtype=np.float)
        old_training_loss = np.zeros((6,), dtype=np.float)
        test_loss = np.zeros((6,), dtype=np.float)
        test_loss_logs = []
        # added by syshin
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch norm.
        test_save_path = os.path.join(self.output_dir,'test')
        if not os.path.isdir(test_save_path):
            os.mkdir(test_save_path)
        for iter in range(max_iters):
            
            timer.tic()
            if iter >= s_start_iter and iter <= s_end_iter:        
                # get one batch
                blobs_train_s = data_layer_train_s.forward()
                
                if self.lr_scheduling=='const':
                    cur_lr = self.lr
                elif self.lr_scheduling=='pc':
                    cur_lr = sess.run([pc_lr])
                    cur_lr = cur_lr[0]
                elif self.lr_scheduling=='rop':
                    cur_lr = rop_handler.get_cur_lr()
    
                # Make one SGD update
                feed_dict={self.net.data: blobs_train_s['data'], self.net.im_info: blobs_train_s['im_info'], \
                           self.net.gt_boxes: blobs_train_s['gt_boxes'], self.net.num_gt_boxes: blobs_train_s['num_gt_boxes'], \
                           self.net.keep_prob: 0.5, self.net.is_training: True, self.net.is_ws: False, self.learning_rate: cur_lr}
    
                run_options = None
                run_metadata = None
                if cfg.TRAIN.DEBUG_TIMELINE:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
    
                rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, _, _ = \
                sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, train_op_s, extra_update_ops],
                            feed_dict=feed_dict,
                            options=run_options,
                            run_metadata=run_metadata)     
            else:
                rpn_loss_cls_value = old_training_loss[1]
                rpn_loss_box_value = old_training_loss[2]
                loss_cls_value = old_training_loss[3]
                loss_box_value = old_training_loss[4]
            
            if iter >= ws_start_iter and iter <= ws_end_iter and (iter+1) % cfg.TRAIN.WS_TRAIN_INTERVAL == 0:  
                # get one batch
                blobs_train_ws = data_layer_train_ws.forward()
                
                if self.lr_scheduling=='const':
                    cur_lr = self.lr
                elif self.lr_scheduling=='pc':
                    cur_lr = sess.run([pc_lr])
                    cur_lr = cur_lr[0]
                elif self.lr_scheduling=='rop':
                    cur_lr = rop_handler.get_cur_lr()
    
                # Make one SGD update
                feed_dict={self.net.data: blobs_train_ws['data'], self.net.im_info: blobs_train_ws['im_info'], \
                           self.net.gt_boxes: blobs_train_ws['gt_boxes'], self.net.num_gt_boxes: blobs_train_ws['num_gt_boxes'], \
                           self.net.keep_prob: 0.5, self.net.is_training: True, self.net.is_ws: True, self.learning_rate: cur_lr}

                run_options = None
                run_metadata = None
                if cfg.TRAIN.DEBUG_TIMELINE:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
       
                mil_loss_cls_value, _ = sess.run([mil_cross_entropy, train_op_ws],
                                                 feed_dict=feed_dict,
                                                 options=run_options,
                                                 run_metadata=run_metadata)                                   
            else:
                if old_training_loss[5] != 0:
                    mil_loss_cls_value = old_training_loss[5]
                else:
                    mil_loss_cls_value = -np.log(1./self.imdb_train_ws.num_classes)
                
            timer.toc() # the time is calculated for one supervised ff & one weakly supervised ff                
            
            # added by syshin
            training_loss += np.array([rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value, \
                                        rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, mil_loss_cls_value])
            # added by syshin

            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(long(time.time() * 1000)) + '-train-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d'%\
                        (iter+1, max_iters)
                print 'total_loss: %.4f'%\
                        (rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value)
                print 'rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, mil_loss_cls: %.4f'%\
                        (rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, mil_loss_cls_value)
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)
            
            if (iter+1) % cfg.TRAIN.TEST_ITERS == 0:
                
                all_boxes = [[[] for _ in xrange(len_test)]
                             for _ in xrange(self.imdb_test.num_classes)]
                for test_idx in xrange(len_test):
                    # get one batch
                    blobs_test = data_layer_test.forward()
        
                    # Make one SGD update
                    feed_dict={self.net.data: blobs_test['data'], self.net.im_info: blobs_test['im_info'], \
                               self.net.gt_boxes: blobs_test['gt_boxes'], self.net.num_gt_boxes: blobs_test['num_gt_boxes'], \
                               self.net.keep_prob: 1.0, self.net.is_training: False, self.net.is_ws: False} # check if 'is_training' is 'True'
                    
                    """rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, mil_loss_cls_value = sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, mil_cross_entropy],
                                                                                                                        feed_dict=feed_dict,
                                                                                                                        options=run_options,
                                                                                                                        run_metadata=run_metadata)"""
                    ## make qualitative results                                                                                                    
                    rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, mil_loss_cls_value, \
                    cls_prob, bbox_pred_, rois = \
                    sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, mil_cross_entropy, \
                              self.net.get_output('cls_prob'), bbox_pred, self.net.get_output('roi-data')[0]],
                              feed_dict=feed_dict,
                              options=run_options,
                              run_metadata=run_metadata)
                           
                    # Apply bounding-box regression deltas
                    temp = skimage.io.imread(self.imdb_test.image_path_at(test_idx))
                    im = np.dstack((temp,temp,temp))
                    boxes = rois[:,1:5] / blobs_test['im_info'][0,2]
                    pred_boxes = clip_boxes(bbox_transform_inv(boxes, bbox_pred_), im.shape)

                    if self.vis:
                        image = im[:, :, (2, 1, 0)]
                        plt.cla()
                        fig = plt.imshow(image, aspect='equal')
                        plt.axis('off')
                        fig.axes.get_xaxis().set_visible(False)
                        fig.axes.get_yaxis().set_visible(False)
                        # draw gt boxes
                        n_gt_bbox = np.sum(self.roidb_test[test_idx]['gt_classes']!=0)
                        for i_gt_bbox in xrange(n_gt_bbox):
                            bbox = self.roidb_test[test_idx]['boxes'][i_gt_bbox,:].astype(np.float32, copy=False)
                            gt_label = self.roidb_test[test_idx]['gt_classes'][i_gt_bbox]
                            plt.gca().add_patch(
                                plt.Rectangle((bbox[0], bbox[1]),
                                              bbox[2] - bbox[0],
                                              bbox[3] - bbox[1], fill=False,
                                              edgecolor=('r' if gt_label==2 else 'b'), linewidth=3)
                                )

                    # skip j = 0, because it's the background class
                    for j in xrange(1, self.imdb_test.num_classes):
                        inds = np.where(cls_prob[:, j] > thresh)[0]
                        cls_scores = cls_prob[inds, j]
                        cls_boxes = pred_boxes[inds, j*4:(j+1)*4]
                        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                            .astype(np.float32, copy=False)
                        keep = nms(cls_dets, cfg.TEST.NMS)
                        cls_dets = cls_dets[keep, :]
                        if self.vis and not cfg.TEST.CLS_AGNOSTIC_NMS:
                            vis_detections(image, self.imdb_test.classes[j], cls_dets)
                        all_boxes[j][test_idx] = cls_dets
                    if cfg.TEST.CLS_AGNOSTIC_NMS:
                        all_dets = np.zeros((0,6), dtype=np.float32)
                        for j in xrange(1, self.imdb_test.num_classes):
                            all_dets = np.concatenate((all_dets, np.hstack((all_boxes[j][test_idx], j*np.ones((all_boxes[j][test_idx].shape[0],1), dtype=np.float32)))), axis=0)
                        keep = nms(all_dets, cfg.TEST.NMS)
                        all_dets = all_dets[keep, :]
                        for j in xrange(1, self.imdb_testnum_classes):
                            inds = np.where(all_dets[:,5]==j)[0]
                            cls_dets = all_dets[inds,:5]
                            all_boxes[j][test_idx] = cls_dets
                            if self.vis:
                                vis_detections(image, self.imdb_test.classes[j], cls_dets)
                    
                    if self.vis:
                        #plt.show()
                        plt.savefig(os.path.join(test_save_path, os.path.splitext(os.path.basename(self.imdb_test.image_path_at(test_idx)))[0]+'.png'), bbox_inches='tight', pad_inches=0)  
                    # Limit to max_per_image detections *over all classes*
                    if max_per_image > 0:
                        image_scores = np.hstack([all_boxes[j][test_idx][:, -1]
                                                  for j in xrange(1, self.imdb_test.num_classes)])
                        if len(image_scores) > max_per_image:
                            image_thresh = np.sort(image_scores)[-max_per_image]
                            for j in xrange(1, self.imdb_test.num_classes):
                                keep = np.where(all_boxes[j][test_idx][:, -1] >= image_thresh)[0]
                                all_boxes[j][test_idx] = all_boxes[j][test_idx][keep, :]
            
                    print 'im_detect: {:d}/{:d}'.format(test_idx+1, len_test)
                    ## make qualitative results
                
                    test_loss += np.array([rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value, \
                                            rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, mil_loss_cls_value])
                                            
                det_file = os.path.join(test_save_path, 'detections.pkl')
                with open(det_file, 'wb') as f:
                    cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
            
                print 'Evaluating detections'
                corloc_list = self.imdb_test.evaluate_detections(all_boxes, test_save_path, '{:d}'.format(iter+1))
                
                training_loss /= cfg.TRAIN.TEST_ITERS
                test_loss /= len_test
                
                test_loss_logs.append(float(test_loss[0]))
                if self.lr_scheduling=='rop':
                    rop_handler.on_val_end(test_loss_logs)
                
                summary = tf.Summary()
                summary.value.add(tag="training_loss_total", simple_value=float(training_loss[0]))
                summary.value.add(tag="training_loss_rpn_loss_cls", simple_value=float(training_loss[1]))
                summary.value.add(tag="training_loss_rpn_loss_box", simple_value=float(training_loss[2]))
                summary.value.add(tag="training_loss_loss_cls", simple_value=float(training_loss[3]))
                summary.value.add(tag="training_loss_loss_box", simple_value=float(training_loss[4]))
                summary.value.add(tag="training_loss_mil_loss_cls", simple_value=float(training_loss[5]))
                summary.value.add(tag="test_loss_total", simple_value=float(test_loss[0]))
                summary.value.add(tag="test_loss_rpn_loss_cls", simple_value=float(test_loss[1]))
                summary.value.add(tag="test_loss_rpn_loss_box", simple_value=float(test_loss[2]))
                summary.value.add(tag="test_loss_loss_cls", simple_value=float(test_loss[3]))
                summary.value.add(tag="test_loss_loss_box", simple_value=float(test_loss[4]))
                summary.value.add(tag="test_loss_mil_loss_cls", simple_value=float(test_loss[5]))
                summary.value.add(tag="corloc for benign", simple_value=float(corloc_list[0]))
                summary.value.add(tag="corloc for malignant", simple_value=float(corloc_list[1]))
                summary.value.add(tag="corloc", simple_value=float(corloc_list[2]))
                summary.value.add(tag="lr", simple_value=float(cur_lr))
                summary_writer.add_summary(summary, global_step=iter+1)
                summary_writer.flush()
                
                print 'iter: %d / %d'%\
                        (iter+1, max_iters)
                print 'training loss'
                print 'total_loss: %.4f'%\
                        (training_loss[0])
                print 'rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, mil_loss_cls: %.4f'%\
                        (training_loss[1], training_loss[2], training_loss[3], training_loss[4], training_loss[5])
                print 'test loss'
                print 'total_loss: %.4f'%\
                        (test_loss[0])
                print 'rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, mil_loss_cls: %.4f'%\
                        (test_loss[1], test_loss[2], test_loss[3], test_loss[4], test_loss[5])
                print 'corloc for benign: %.4f, corloc for malignant: %.4f, corloc: %.4f'%\
                        (corloc_list[0], corloc_list[1], corloc_list[2])
                print 'lr: %.8f'%\
                        (cur_lr)
                
                f_log.write('iter: '+str(iter+1)+' / '+str(max_iters)+'\n')
                f_log.write('training loss'+'\n')
                f_log.write('total_loss: '+str(training_loss[0])+'\n')
                f_log.write('rpn_loss_cls: '+str(training_loss[1])+'\trpn_loss_box: '+str(training_loss[2])+'\tloss_cls: '+str(training_loss[3])+'\tloss_box: '+str(training_loss[4])+'\n'+\
                            'mil_loss_cls: '+str(training_loss[5])+'\n')
                f_log.write('test loss'+'\n')
                f_log.write('total_loss: '+str(test_loss[0])+'\n')
                f_log.write('rpn_loss_cls: '+str(test_loss[1])+'\trpn_loss_box: '+str(test_loss[2])+'\tloss_cls: '+str(test_loss[3])+'\tloss_box: '+str(test_loss[4])+'\n'+\
                            'mil_loss_cls: '+str(test_loss[5])+'\n')
                f_log.write('lr: '+str(cur_lr)+'\n')
                f_log.flush()
                
                old_training_loss = training_loss
                training_loss = np.zeros((6,), dtype=np.float)
                test_loss = np.zeros((6,), dtype=np.float)   

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)
            
        f_log.close()


    def train_model(self, sess, max_iters, s_start_iter, s_end_iter, ws_start_iter, ws_end_iter, max_per_image=300, thresh=0.05):
        """Network training loop."""
        
        global_step = tf.Variable(0, trainable=False)

        data_layer_train = get_data_layer([self.roidb_train_s,self.roidb_train_ws], self.net_name, self.imdb_train_s.num_classes, is_training=True, is_ws=True, is_joint=True)  
        data_layer_test = get_data_layer(self.roidb_test, self.net_name, self.imdb_test.num_classes, is_training=False)

        ## RPN
        # classification loss
        rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score_reshape'),[-1,2])
        rpn_label = tf.reshape(self.net.get_output('rpn-data')[0],[-1])
        # ignore_label(-1)
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label,-1))),[-1,2])
        rpn_label = tf.reshape(tf.gather(rpn_label,tf.where(tf.not_equal(rpn_label,-1))),[-1])
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

        # bounding box regression L1 loss               
        rpn_bbox_pred = tf.slice(self.net.get_output('rpn_bbox_pred'),[0,0,0,0],[cfg.TRAIN.IMS_PER_BATCH,-1,-1,-1])
        rpn_bbox_targets = tf.slice(tf.transpose(self.net.get_output('rpn-data')[1],[0,2,3,1]),[0,0,0,0],[cfg.TRAIN.IMS_PER_BATCH,-1,-1,-1])
        rpn_bbox_inside_weights = tf.slice(tf.transpose(self.net.get_output('rpn-data')[2],[0,2,3,1]),[0,0,0,0],[cfg.TRAIN.IMS_PER_BATCH,-1,-1,-1])
        rpn_bbox_outside_weights = tf.slice(tf.transpose(self.net.get_output('rpn-data')[3],[0,2,3,1]),[0,0,0,0],[cfg.TRAIN.IMS_PER_BATCH,-1,-1,-1])
        smoothL1_sign = tf.cast(tf.less(tf.abs(tf.subtract(rpn_bbox_pred, rpn_bbox_targets)),1),tf.float32)
        rpn_loss_box = tf.multiply(tf.reduce_mean(tf.reduce_sum(tf.multiply(rpn_bbox_outside_weights,tf.add(
                       tf.multiply(tf.multiply(tf.pow(tf.multiply(rpn_bbox_inside_weights, tf.subtract(rpn_bbox_pred, rpn_bbox_targets))*3,2),0.5),smoothL1_sign),
                       tf.multiply(tf.subtract(tf.abs(tf.subtract(rpn_bbox_pred, rpn_bbox_targets)),0.5/9.0),tf.abs(smoothL1_sign-1)))), axis=[1,2])),10)                   
           
        ## R-CNN
        # classification loss        
        cls_score = self.net.get_output('cls_score')
        label = tf.reshape(self.net.get_output('roi-data')[1],[-1])
        len_valid = tf.reshape(tf.size(label),[1,1])
        size_valid = tf.reshape(tf.concat(axis=0, values=[len_valid,tf.constant(-1,dtype=tf.int32,shape=[1,1])]),[2])
        cls_score_s = tf.slice(cls_score, [0,0], size_valid)
        # simple cross entropy loss # 
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score_s, labels=label))
        # weighted cross entropy loss #
        #temp_weight_s = np.array([1, 1, (1-cfg.TRAIN.S_MAL_PCT)/cfg.TRAIN.S_MAL_PCT], dtype=np.float32)
        #temp_weight_s = [list(temp_weight_s/np.sum(temp_weight_s))]
        #class_weight_s = tf.multiply(3., tf.constant(temp_weight_s))
        """hist = tf.add(1e-10,tf.divide(tf.cast(tf.histogram_fixed_width(tf.cast(label,tf.float64), tf.cast([0,self.imdb_train_s.num_classes],tf.float64), self.imdb_train_s.num_classes), \
                tf.float32), tf.cast(tf.size(label),tf.float32)))
        class_weight_s = tf.reshape(tf.cast(tf.divide(1./self.imdb_train_s.num_classes,hist),tf.float32),[1,self.imdb_train_s.num_classes])
        class_weight_s = tf.multiply(3.,tf.divide(class_weight_s,tf.reduce_sum(class_weight_s)))"""
        #weight_per_label_s = tf.transpose(tf.matmul(tf.one_hot(label, depth=self.imdb_train_s.num_classes),tf.transpose(class_weight_s))) #shape [1,N]
        #cross_entropy = tf.reduce_mean(tf.multiply(weight_per_label_s, tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score_s, labels=label)))
        
        # bounding box regression L1 loss
        bbox_pred = self.net.get_output('bbox_pred')
        bbox_pred = tf.slice(bbox_pred, [0,0], size_valid)  
        bbox_targets = self.net.get_output('roi-data')[2]
        bbox_inside_weights = self.net.get_output('roi-data')[3]
        bbox_outside_weights = self.net.get_output('roi-data')[4]
        loss_box = tf.reduce_mean(tf.reduce_sum(tf.multiply(bbox_outside_weights,tf.multiply(bbox_inside_weights, tf.abs(tf.subtract(bbox_pred, bbox_targets)))), axis=[1]))

        ## MIL
        # classification loss
        begin = tf.reshape(tf.concat(axis=0, values=[len_valid,tf.constant(0,dtype=tf.int32,shape=[1,1])]),[2])
        cls_score_ws = tf.slice(cls_score, begin, [-1,-1])  
        batch_inds = tf.subtract(tf.slice(self.net.get_output('roi-data')[0],begin,[-1,1]), cfg.TRAIN.IMS_PER_BATCH)
        mil_label = tf.cast(tf.reshape(tf.slice(self.net.get_output('im_info'), [cfg.TRAIN.IMS_PER_BATCH,3], [-1,1]),[-1]),tf.int32) # image-level labels
        funcs = [mil_core.get_mal_max_logit, mil_core.get_mal_max_logit]
        #mil_cls_score, scale_factors = mil_core.get_bag_logit(cls_score_ws, batch_inds, self.imdb_train_ws.num_classes, tf.reshape(mil_label,[-1,1]), cfg.TRAIN.WS_IMS_PER_BATCH, funcs)   
        mil_cls_score, _ = mil_core.get_bag_logit(cls_score_ws, batch_inds, self.imdb_train_ws.num_classes, tf.reshape(mil_label,[-1,1]), cfg.TRAIN.WS_IMS_PER_BATCH, funcs)
        scale_factors = tf.tile(tf.reshape(tf.subtract(1.,tf.train.exponential_decay(0.99, global_step, 2000, 0.9, staircase=True)),[-1]), tf.reshape(tf.size(mil_label),[-1])) 
   
        ## simple cross entropy loss
        #mil_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mil_cls_score, labels=mil_label))
        ## weighted cross entropy loss  
        temp_weight_ws = [[0, cfg.TRAIN.WS_MAL_PCT, 1-cfg.TRAIN.WS_MAL_PCT]]
        class_weight_ws = tf.constant(temp_weight_ws)
        weight_per_label_ws = tf.transpose(tf.matmul(tf.one_hot(mil_label, depth=self.imdb_train_ws.num_classes),tf.transpose(class_weight_ws))) #shape [1, batch_size]
        # this is the weight for each datapoint, depending on its label   
        if cfg.TRAIN.WS_LOSS_USE_ADAPTIVE_SCALE_FACTOR:
            mil_cross_entropy = tf.reduce_mean(tf.multiply(scale_factors, tf.multiply(weight_per_label_ws, tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mil_cls_score, labels=mil_label))))   
        else:
            mil_cross_entropy = tf.reduce_mean(tf.multiply(cfg.TRAIN.WS_LOSS_SCALE_FACTOR, tf.multiply(weight_per_label_ws, tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mil_cls_score, labels=mil_label))))
        
        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box        
        
        # added by syshin
        weights_only = filter( lambda x: x.name.endswith('weights:0'), tf.trainable_variables() )
        weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only])) * cfg.TRAIN.WEIGHT_DECAY
        loss += weight_decay
        # added by syshin
        
        # learning rate
        if self.lr_scheduling=='const':
            pass
        elif self.lr_scheduling=='pc':
            boundaries = [int(max_iters*0.75)]
            values = [self.lr,self.lr*0.1]
            pc_lr = tf.train.piecewise_constant(global_step, boundaries, values)            
        elif self.lr_scheduling=='rop':
            rop_handler = ReduceLROnPlateau(self.lr, monitor='val_loss', factor=0.5, patience=5, verbose=True, mode='min', epsilon=1e-03, cooldown=0, min_lr=0)
        else:
            NotImplemented
        
        # optimizer    
        if self.opt=='adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=0.1)
        elif self.opt=='sgd':
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, cfg.TRAIN.MOMENTUM, use_nesterov=True)  
        else:
            NotImplemented

        grads_and_vars_s = optimizer.compute_gradients(loss)
        grads_and_vars_ws = optimizer.compute_gradients(mil_cross_entropy)                                       
        grads_and_vars = map(lambda gv_tuple: (add_tensors_wo_none([gv_tuple[0][0],gv_tuple[1][0]]), gv_tuple[0][1]), list(zip(grads_and_vars_s,grads_and_vars_ws)))                                            

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        summary_writer = tf.summary.FileWriter(self.output_dir, sess.graph) # added by syshin

        # initialize variables
        sess.run(tf.global_variables_initializer())        
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)

        last_snapshot_iter = -1
        timer = Timer()
        # added by syshin
        f_log = open(os.path.join(self.output_dir,'log.txt'), 'w')
        len_training_s = len(self.roidb_train_s) # length of the augmented tranining set
        len_training_ws = len(self.roidb_train_ws) # length of the augmented tranining set
        len_test = len(self.roidb_test)
        training_loss = np.zeros((6,), dtype=np.float)
        old_training_loss = np.zeros((6,), dtype=np.float)
        test_loss = np.zeros((6,), dtype=np.float)
        test_loss_logs = []
        # added by syshin
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch norm.
        test_save_path = os.path.join(self.output_dir,'test')
        if not os.path.isdir(test_save_path):
            os.mkdir(test_save_path)
        for iter in range(max_iters):
            
            timer.tic()
            
            # get one batch
            blobs_train = data_layer_train.forward()

            if self.lr_scheduling=='const':
                cur_lr = self.lr
            elif self.lr_scheduling=='pc':
                cur_lr = sess.run([pc_lr])
                cur_lr = cur_lr[0]
            elif self.lr_scheduling=='rop':
                cur_lr = rop_handler.get_cur_lr()

            # Make one SGD update
            feed_dict={self.net.data: blobs_train['data'], self.net.im_info: blobs_train['im_info'], \
                       self.net.gt_boxes: blobs_train['gt_boxes'], self.net.num_gt_boxes: blobs_train['num_gt_boxes'], \
                       self.net.keep_prob: 0.5, self.net.is_training: True, self.net.is_ws: False, self.learning_rate: cur_lr}

            run_options = None
            run_metadata = None
            if cfg.TRAIN.DEBUG_TIMELINE:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            
            rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, \
            mil_loss_cls_value, _, _ = \
            sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, \
                      mil_cross_entropy, train_op, extra_update_ops],
                      feed_dict=feed_dict,
                      options=run_options,
                      run_metadata=run_metadata)

            timer.toc()          
            
            # added by syshin
            training_loss += np.array([rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value, \
                                        rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, mil_loss_cls_value])
            # added by syshin

            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(long(time.time() * 1000)) + '-train-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d'%\
                        (iter+1, max_iters)
                print 'total_loss: %.4f'%\
                        (rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value)
                print 'rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, mil_loss_cls: %.4f'%\
                        (rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, mil_loss_cls_value)
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)
            
            if (iter+1) % cfg.TRAIN.TEST_ITERS == 0:
                
                all_boxes = [[[] for _ in xrange(len_test)]
                             for _ in xrange(self.imdb_test.num_classes)]
                for test_idx in xrange(len_test):
                    # get one batch
                    blobs_test = data_layer_test.forward()
        
                    # Make one SGD update
                    feed_dict={self.net.data: blobs_test['data'], self.net.im_info: blobs_test['im_info'], \
                               self.net.gt_boxes: blobs_test['gt_boxes'], self.net.num_gt_boxes: blobs_test['num_gt_boxes'], \
                               self.net.keep_prob: 1.0, self.net.is_training: False, self.net.is_ws: False} # check if 'is_training' is 'True'
                    
                    """rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, mil_loss_cls_value = sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, mil_cross_entropy],
                                                                                                                        feed_dict=feed_dict,
                                                                                                                        options=run_options,
                                                                                                                        run_metadata=run_metadata)"""
                    ## make qualitative results                                                                                                    
                    rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, \
                    cls_prob, bbox_pred_, rois = \
                    sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, \
                              self.net.get_output('cls_prob'), bbox_pred, self.net.get_output('roi-data')[0]],
                              feed_dict=feed_dict,
                              options=run_options,
                              run_metadata=run_metadata)
                           
                    # Apply bounding-box regression deltas
                    temp = skimage.io.imread(self.imdb_test.image_path_at(test_idx))
                    im = np.dstack((temp,temp,temp))
                    boxes = rois[:,1:5] / blobs_test['im_info'][0,2]
                    pred_boxes = clip_boxes(bbox_transform_inv(boxes, bbox_pred_), im.shape)

                    if self.vis:
                        image = im[:, :, (2, 1, 0)]
                        plt.cla()
                        fig = plt.imshow(image, aspect='equal')
                        plt.axis('off')
                        fig.axes.get_xaxis().set_visible(False)
                        fig.axes.get_yaxis().set_visible(False)
                        # draw gt boxes
                        n_gt_bbox = np.sum(self.roidb_test[test_idx]['gt_classes']!=0)
                        for i_gt_bbox in xrange(n_gt_bbox):
                            bbox = self.roidb_test[test_idx]['boxes'][i_gt_bbox,:].astype(np.float32, copy=False)
                            gt_label = self.roidb_test[test_idx]['gt_classes'][i_gt_bbox]
                            plt.gca().add_patch(
                                plt.Rectangle((bbox[0], bbox[1]),
                                              bbox[2] - bbox[0],
                                              bbox[3] - bbox[1], fill=False,
                                              edgecolor=('r' if gt_label==2 else 'b'), linewidth=3)
                                )

                    # skip j = 0, because it's the background class
                    for j in xrange(1, self.imdb_test.num_classes):
                        inds = np.where(cls_prob[:, j] > thresh)[0]
                        cls_scores = cls_prob[inds, j]
                        cls_boxes = pred_boxes[inds, j*4:(j+1)*4]
                        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                            .astype(np.float32, copy=False)
                        keep = nms(cls_dets, cfg.TEST.NMS)
                        cls_dets = cls_dets[keep, :]
                        if self.vis and not cfg.TEST.CLS_AGNOSTIC_NMS:
                            vis_detections(image, self.imdb_test.classes[j], cls_dets)
                        all_boxes[j][test_idx] = cls_dets
                    if cfg.TEST.CLS_AGNOSTIC_NMS:
                        all_dets = np.zeros((0,6), dtype=np.float32)
                        for j in xrange(1, self.imdb_test.num_classes):
                            all_dets = np.concatenate((all_dets, np.hstack((all_boxes[j][test_idx], j*np.ones((all_boxes[j][test_idx].shape[0],1), dtype=np.float32)))), axis=0)
                        keep = nms(all_dets, cfg.TEST.NMS)
                        all_dets = all_dets[keep, :]
                        for j in xrange(1, self.imdb_testnum_classes):
                            inds = np.where(all_dets[:,5]==j)[0]
                            cls_dets = all_dets[inds,:5]
                            all_boxes[j][test_idx] = cls_dets
                            if self.vis:
                                vis_detections(image, self.imdb_test.classes[j], cls_dets)
                    
                    if self.vis:
                        #plt.show()
                        plt.savefig(os.path.join(test_save_path, os.path.splitext(os.path.basename(self.imdb_test.image_path_at(test_idx)))[0]+'.png'), bbox_inches='tight', pad_inches=0)  
                    # Limit to max_per_image detections *over all classes*
                    if max_per_image > 0:
                        image_scores = np.hstack([all_boxes[j][test_idx][:, -1]
                                                  for j in xrange(1, self.imdb_test.num_classes)])
                        if len(image_scores) > max_per_image:
                            image_thresh = np.sort(image_scores)[-max_per_image]
                            for j in xrange(1, self.imdb_test.num_classes):
                                keep = np.where(all_boxes[j][test_idx][:, -1] >= image_thresh)[0]
                                all_boxes[j][test_idx] = all_boxes[j][test_idx][keep, :]
            
                    print 'im_detect: {:d}/{:d}'.format(test_idx+1, len_test)
                    ## make qualitative results
                
                    test_loss += np.array([rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value, \
                                            rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, 0])
                                            
                det_file = os.path.join(test_save_path, 'detections.pkl')
                with open(det_file, 'wb') as f:
                    cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
            
                print 'Evaluating detections'
                corloc_list = self.imdb_test.evaluate_detections(all_boxes, test_save_path, '{:d}'.format(iter+1))
                
                training_loss /= cfg.TRAIN.TEST_ITERS
                test_loss /= len_test
                
                test_loss_logs.append(float(test_loss[0]))
                if self.lr_scheduling=='rop':
                    rop_handler.on_val_end(test_loss_logs)
                
                summary = tf.Summary()
                summary.value.add(tag="training_loss_total", simple_value=float(training_loss[0]))
                summary.value.add(tag="training_loss_rpn_loss_cls", simple_value=float(training_loss[1]))
                summary.value.add(tag="training_loss_rpn_loss_box", simple_value=float(training_loss[2]))
                summary.value.add(tag="training_loss_loss_cls", simple_value=float(training_loss[3]))
                summary.value.add(tag="training_loss_loss_box", simple_value=float(training_loss[4]))
                summary.value.add(tag="training_loss_mil_loss_cls", simple_value=float(training_loss[5]))
                summary.value.add(tag="test_loss_total", simple_value=float(test_loss[0]))
                summary.value.add(tag="test_loss_rpn_loss_cls", simple_value=float(test_loss[1]))
                summary.value.add(tag="test_loss_rpn_loss_box", simple_value=float(test_loss[2]))
                summary.value.add(tag="test_loss_loss_cls", simple_value=float(test_loss[3]))
                summary.value.add(tag="test_loss_loss_box", simple_value=float(test_loss[4]))
                summary.value.add(tag="test_loss_mil_loss_cls", simple_value=float(test_loss[5]))
                summary.value.add(tag="corloc for benign", simple_value=float(corloc_list[0]))
                summary.value.add(tag="corloc for malignant", simple_value=float(corloc_list[1]))
                summary.value.add(tag="corloc", simple_value=float(corloc_list[2]))
                summary.value.add(tag="lr", simple_value=float(cur_lr))
                summary_writer.add_summary(summary, global_step=iter+1)
                summary_writer.flush()
                
                print 'iter: %d / %d'%\
                        (iter+1, max_iters)
                print 'training loss'
                print 'total_loss: %.4f'%\
                        (training_loss[0])
                print 'rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, mil_loss_cls: %.4f'%\
                        (training_loss[1], training_loss[2], training_loss[3], training_loss[4], training_loss[5])
                print 'test loss'
                print 'total_loss: %.4f'%\
                        (test_loss[0])
                print 'rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, mil_loss_cls: %.4f'%\
                        (test_loss[1], test_loss[2], test_loss[3], test_loss[4], test_loss[5])
                print 'corloc for benign: %.4f, corloc for malignant: %.4f, corloc: %.4f'%\
                        (corloc_list[0], corloc_list[1], corloc_list[2])
                print 'lr: %.8f'%\
                        (cur_lr)
                
                f_log.write('iter: '+str(iter+1)+' / '+str(max_iters)+'\n')
                f_log.write('training loss'+'\n')
                f_log.write('total_loss: '+str(training_loss[0])+'\n')
                f_log.write('rpn_loss_cls: '+str(training_loss[1])+'\trpn_loss_box: '+str(training_loss[2])+'\tloss_cls: '+str(training_loss[3])+'\tloss_box: '+str(training_loss[4])+'\n'+\
                            'mil_loss_cls: '+str(training_loss[5])+'\n')
                f_log.write('test loss'+'\n')
                f_log.write('total_loss: '+str(test_loss[0])+'\n')
                f_log.write('rpn_loss_cls: '+str(test_loss[1])+'\trpn_loss_box: '+str(test_loss[2])+'\tloss_cls: '+str(test_loss[3])+'\tloss_box: '+str(test_loss[4])+'\n'+\
                            'mil_loss_cls: '+str(test_loss[5])+'\n')
                f_log.write('lr: '+str(cur_lr)+'\n')
                f_log.flush()
                
                old_training_loss = training_loss
                training_loss = np.zeros((6,), dtype=np.float)
                test_loss = np.zeros((6,), dtype=np.float)   

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)
            
        f_log.close()


def add_tensors_wo_none(tensor_list):
    """Adds all input tensors element-wise while filtering out none tensors."""
    #print tensor_list
    return tf.add_n(filter(lambda x: (x is not None), tensor_list))


def vis_pseudo_gt(data, gt_boxes, pseudo_gt, output_dir):
    assert data.shape[0] == (cfg.TRAIN.IMS_PER_BATCH+cfg.TRAIN.WS_IMS_PER_BATCH)
    for i in xrange(cfg.TRAIN.WS_IMS_PER_BATCH):
        
        batch_idx = i+cfg.TRAIN.IMS_PER_BATCH
        image = data[batch_idx,:,:,:]
        image = (image-np.amin(image))/(np.amax(image)-np.amin(image))
        
        for j in xrange(cfg.TRAIN.OR_N):
        
            plt.cla()
            fig = plt.imshow(image, aspect='equal')
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            # draw gt boxes
            n_gt_bbox = np.sum(gt_boxes[batch_idx,:,-1]!=0)
            for k in xrange(n_gt_bbox):
                bbox = gt_boxes[batch_idx,k,:4].astype(np.float32, copy=False)                    
                gt_label = gt_boxes[batch_idx,k,-1]
                plt.gca().add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor=('r' if gt_label==2 else 'b'), linewidth=3)
                    )
            # draw pseudo gt boxes
            t_gt_boxes_pseudo = pseudo_gt[j][0]
            t_num_gt_boxes_pseudo = pseudo_gt[j][1]
            t_max_probs_pseudo = pseudo_gt[j][2]
            for k in xrange(t_num_gt_boxes_pseudo[i]):
                bbox = t_gt_boxes_pseudo[i,k,:4]
                class_name = 'malignant' if t_gt_boxes_pseudo[i,k,-1]==2 else 'benign' 
                score = t_max_probs_pseudo[i]
                plt.gca().add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor=('r' if class_name=='malignant' else 'b'), linewidth=3, linestyle='dashed')
                    )
                plt.gca().text(bbox[0], bbox[1] + 20,
                     '{:s} {:.3f}'.format(class_name, score),
                     bbox=dict(facecolor=('red' if class_name=='malignant' else 'blue'), alpha=0.5),
                     fontsize=14, color='white')
                     
            #plt.show()
            plt.savefig(output_dir+'_{}_{}.png'.format(i,j+1), bbox_inches='tight', pad_inches=0)
            plt.close()


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            #gdl_roidb.prepare_roidb(imdb)
            raise NotImplementedError
        else:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb


def get_data_layer(roidbs, net_name, num_classes, is_training, is_ws=False, is_joint=False):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            #layer = GtDataLayer(roidb) # the argument 'is_training' is not considered here.
            raise NotImplementedError # 190921
        else:
            if is_joint:
                layer = RoIDataLayerJoint(roidbs[0], roidbs[1], net_name, num_classes, is_training)
            else:
                layer = RoIDataLayer(roidbs, net_name, num_classes, is_training, is_ws)

    else:
        layer = RoIDataLayer(roidb, net_name, num_classes, is_training)

    return layer


def train_net_alter(network, imdb_train_s, roidb_train_s, imdb_train_ws, roidb_train_ws, imdb_test, roidb_test, \
                    output_dir, pretrained_model=None, \
                    max_iters=80000, s_start_iter=0, s_end_iter=80000, ws_start_iter=0, ws_end_iter=80000, \
                    opt='adam', lr=5e-04, lr_scheduling='const', vis=False):
    """Train a Faster R-CNN using alternating mini-batches each from supervised and weakly supervised sets"""

    #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imdb_train_s, roidb_train_s, imdb_train_ws, roidb_train_ws, \
                           imdb_test, roidb_test, output_dir, pretrained_model=pretrained_model, \
                           opt=opt, lr=lr, lr_scheduling=lr_scheduling, vis=vis)
        print 'Solving...'
        sw.train_model_alter(sess, max_iters, s_start_iter, s_end_iter, ws_start_iter, ws_end_iter)
        print 'done solving'


def train_net(network, imdb_train_s, roidb_train_s, imdb_train_ws, roidb_train_ws, imdb_test, roidb_test, \
              output_dir, pretrained_model=None, \
              max_iters=80000, s_start_iter=0, s_end_iter=80000, ws_start_iter=0, ws_end_iter=80000, \
              opt='adam', lr=5e-04, lr_scheduling='const', vis=False):
    """Train a Faster R-CNN using combined mini-batches from supervised and weakly supervised sets"""

    #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imdb_train_s, roidb_train_s, imdb_train_ws, roidb_train_ws, \
                           imdb_test, roidb_test, output_dir, pretrained_model=pretrained_model, \
                           opt=opt, lr=lr, lr_scheduling=lr_scheduling, vis=vis)
        print 'Solving...'
        sw.train_model(sess, max_iters, s_start_iter, s_end_iter, ws_start_iter, ws_end_iter)
        print 'done solving'