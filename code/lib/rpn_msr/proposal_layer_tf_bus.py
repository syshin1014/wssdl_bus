# modified by syshin
# the original code is 'proposal_layer_tf.py'

import numpy as np
import yaml
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms
import pdb
from utils.timer import Timer

DEBUG = False
"""
Outputs object detection proposals by applying estimated bounding-box
transformations to a set of regular boxes (called "anchors").
"""

def proposal_layer(rpn_cls_prob_reshape,rpn_bbox_pred,im_info,is_training,is_ws,_feat_stride = [16,],anchor_scales = [8, 16, 32]):
    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    #layer_params = yaml.load(self.param_str_) 

    rpn_cls_prob_reshape = np.transpose(rpn_cls_prob_reshape,[0,3,1,2])
    rpn_bbox_pred = np.transpose(rpn_bbox_pred,[0,3,1,2])
    #rpn_cls_prob_reshape = np.transpose(np.reshape(rpn_cls_prob_reshape,[1,rpn_cls_prob_reshape.shape[0],rpn_cls_prob_reshape.shape[1],rpn_cls_prob_reshape.shape[2]]),[0,3,2,1])
    #rpn_bbox_pred = np.transpose(rpn_bbox_pred,[0,3,2,1])
    
    batch_size = im_info.shape[0]
    #cfg_key = 'TEST'
    cfg_key = 'TRAIN' if is_training else 'TEST'
    
    pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
    min_size      = cfg[cfg_key].RPN_MIN_SIZE

    #### same for all batch element ####
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]
    
    height, width = rpn_cls_prob_reshape.shape[-2:]
    
    # Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    anchors = _anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))
    #### same for all batch element ####       
 
    blob = np.zeros((0,5), dtype=np.float32)
    for i in xrange(batch_size): 
    
        t_im_info = im_info[i,:]
        t_rpn_cls_prob_reshape = rpn_cls_prob_reshape[[i],:,:,:]
        t_rpn_bbox_pred = rpn_bbox_pred[[i],:,:,:]
    
        assert t_rpn_cls_prob_reshape.shape[0] == 1, \
            'Only single item batches are supported'
    
        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = t_rpn_cls_prob_reshape[:, _num_anchors:, :, :]
        bbox_deltas = t_rpn_bbox_pred
        #im_info = bottom[2].data[0, :]
    
        if DEBUG:
            print 'im_size: ({}, {})'.format(t_im_info[0], t_im_info[1])
            print 'scale: {}'.format(t_im_info[2])
        
        if DEBUG:
            print 'score map size: {}'.format(scores.shape)
    
        # 1. Generate proposals from bbox deltas and shifted anchors    

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
    
        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
    
        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)
    
        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, t_im_info[:2])
    
        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * t_im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]
    
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]
    
        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]
        # Output rois blob
        batch_inds = np.ones((proposals.shape[0], 1), dtype=np.float32)*i
        t_blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        blob = np.concatenate((blob,t_blob))
        
    return blob


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep