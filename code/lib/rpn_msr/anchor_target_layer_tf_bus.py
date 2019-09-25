# modified by syshin
# the original code is 'anchor_target_layer_tf.py'

import os
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from utils.cython_bbox_ui import bbox_overlaps_ui
from fast_rcnn.bbox_transform import bbox_transform
import pdb


DEBUG = False


def anchor_target_layer(rpn_cls_score, gt_boxes, num_gt_boxes, im_info, data, _feat_stride=[16,], anchor_scales=[4 ,8, 16, 32], dataset='SNUBH'):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]

    if DEBUG:
        print 'anchors:'
        print _anchors
        print 'anchor shapes:'
        print np.hstack(( 
            _anchors[:, 2::4] - _anchors[:, 0::4],
            _anchors[:, 3::4] - _anchors[:, 1::4],
        ))
        _counts = cfg.EPS
        _sums = np.zeros((1, 4))
        _squared_sums = np.zeros((1, 4))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0

    # allow boxes to sit over the edge by a small amount
    _allowed_border =  0

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]
    batch_size = rpn_cls_score.shape[0]
    
    #### same for all batch element ####
    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)
    #### same for all batch element ####    
    
    rpn_labels = np.zeros((0, 1, A * height, width), dtype=np.float32)
    rpn_bbox_targets = np.zeros((0, A * 4, height, width), dtype=np.float32)
    rpn_bbox_inside_weights = np.zeros((0, A * 4, height, width), dtype=np.float32)
    rpn_bbox_outside_weights = np.zeros((0, A * 4, height, width), dtype=np.float32)
    for i in xrange(batch_size):
        
        t_im_info = im_info[i,:]
        t_rpn_cls_score = rpn_cls_score[[i],:,:,:]
        t_num_gt_boxes =  num_gt_boxes[i]
        t_gt_boxes = gt_boxes[i,:t_num_gt_boxes,:]
    
        assert t_rpn_cls_score.shape[0] == 1, \
            'Only single item batches are supported'

        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width
            print ''
            print 'im_size: ({}, {})'.format(t_im_info[0], t_im_info[1])
            print 'scale: {}'.format(t_im_info[2])
            print 'height, width: ({}, {})'.format(height, width)
            print 'rpn: gt_boxes.shape', t_gt_boxes.shape
            print 'rpn: gt_boxes', t_gt_boxes
    
        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -_allowed_border) &
            (all_anchors[:, 1] >= -_allowed_border) &
            (all_anchors[:, 2] < t_im_info[1] + _allowed_border) &  # width
            (all_anchors[:, 3] < t_im_info[0] + _allowed_border)    # height
        )[0]
    
        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', len(inds_inside)
    
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print 'anchors.shape', anchors.shape
    
        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)
        
        if dataset=='SNUBH':
            ################## SNUBH dataset ##################
            # 'gt_boxes' contains boxes not only for fg but also for bg
            # while 'gt_boxes' in 'pascal voc' only has boxes for fg 
            b_pos = np.transpose(t_gt_boxes[:,4]!=0) ## added by syshin
            num_pos = np.sum(b_pos) ## added by syshin
            exist_neg = (t_gt_boxes.shape[0] != num_pos)
        
            # overlaps between the anchors and the gt boxes
            # overlaps (ex, gt)
            overlaps_pos = bbox_overlaps(
                    np.ascontiguousarray(anchors, dtype=np.float),
                    np.ascontiguousarray(t_gt_boxes[:num_pos,:], dtype=np.float))
                
            argmax_overlaps_pos = overlaps_pos.argmax(axis=1)
            max_overlaps_pos = overlaps_pos[np.arange(len(inds_inside)), argmax_overlaps_pos]
            gt_argmax_overlaps_pos = overlaps_pos.argmax(axis=0)
            gt_max_overlaps_pos = overlaps_pos[gt_argmax_overlaps_pos,
                                       np.arange(overlaps_pos.shape[1])]
            gt_argmax_overlaps_pos = np.where(overlaps_pos == gt_max_overlaps_pos)[0]
            
            if exist_neg:
                overlaps_neg = bbox_overlaps_ui(
                        np.ascontiguousarray(anchors, dtype=np.float),
                        np.ascontiguousarray(t_gt_boxes[num_pos:,:], dtype=np.float))
                
                argmax_overlaps_neg = overlaps_neg.argmax(axis=1)
                max_overlaps_neg = overlaps_neg[np.arange(len(inds_inside)), argmax_overlaps_neg]
        
            if not cfg.TRAIN.RPN_CLOBBER_POSITIVES and exist_neg:
                # assign bg labels first so that positive labels can clobber them
                labels[max_overlaps_neg >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 0
        
            # fg label: for each gt, anchor with highest overlap
            labels[gt_argmax_overlaps_pos] = 1
        
            # fg label: above threshold IOU
            labels[max_overlaps_pos >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
            ################## SNUBH dataset ##################
        else:
            if dataset=='SNUBH_FG':
                ################## SNUBH dataset(w/o bg boxes) ##################
                # 'gt_boxes' contains boxes not only for fg but also for bg
                # but, here we use only the fg boxes
                b_pos = np.transpose(t_gt_boxes[:,4]!=0)
                num_pos = np.sum(b_pos)
                t_gt_boxes = t_gt_boxes[:num_pos,:]
                ################## SNUBH dataset(w/o bg boxes) ##################
            
            ################## UDIAT dataset ##################
            # 'gt_boxes' has boxes only for fg
            # no action is required
            ################## UDIAT dataset ##################
    
            # overlaps between the anchors and the gt boxes
            # overlaps (ex, gt)
            overlaps = bbox_overlaps(
                    np.ascontiguousarray(anchors, dtype=np.float),
                    np.ascontiguousarray(t_gt_boxes, dtype=np.float))
            
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                       np.arange(overlaps.shape[1])]
            gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
            
            if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels first so that positive labels can clobber them
                labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    
            # fg label: for each gt, anchor with highest overlap
            labels[gt_argmax_overlaps] = 1
        
            # fg label: above threshold IOU
            labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
        
            if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels last so that negative labels can clobber positives
                labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    
        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1
    
        # subsample negative labels if we have too many
        # num_bg = batch_size - possible_num_fg
        # this may result in the class imbalance in batches
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            #print "was %s inds, disabling %s, now %s inds" % (
                #len(bg_inds), len(disable_inds), np.sum(labels == 0))
        
        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)        
        if dataset=='SNUBH':
            bbox_targets = _compute_targets(anchors, t_gt_boxes[argmax_overlaps_pos, :])
        else:
            bbox_targets = _compute_targets(anchors, t_gt_boxes[argmax_overlaps, :])
    
        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)
    
        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights
    
        if DEBUG:
            _sums += bbox_targets[labels == 1, :].sum(axis=0)
            _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            _counts += np.sum(labels == 1)
            means = _sums / _counts
            stds = np.sqrt(_squared_sums / _counts - means ** 2)
            print 'means:'
            print means
            print 'stdevs:'
            print stds
    
        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)
    
        if DEBUG:
            print 'rpn: max max_overlap_pos', np.max(max_overlaps_pos)
            if exist_neg:
                print 'rpn: max max_overlap_neg', np.max(max_overlaps_neg)
            #print 'rpn: max max_overlap', np.max(max_overlaps)
            print 'rpn: num_positive', np.sum(labels == 1)
            print 'rpn: num_negative', np.sum(labels == 0)
            _fg_sum += np.sum(labels == 1)
            _bg_sum += np.sum(labels == 0)
            _count += 1
            print 'rpn: num_positive avg', _fg_sum / _count
            print 'rpn: num_negative avg', _bg_sum / _count
    
        # labels
        #pdb.set_trace()
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        rpn_labels = np.concatenate((rpn_labels,labels))
    
        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    
        rpn_bbox_targets = np.concatenate((rpn_bbox_targets,bbox_targets))
        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        #assert bbox_inside_weights.shape[2] == height
        #assert bbox_inside_weights.shape[3] == width
    
        rpn_bbox_inside_weights = np.concatenate((rpn_bbox_inside_weights,bbox_inside_weights))
    
        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        #assert bbox_outside_weights.shape[2] == height
        #assert bbox_outside_weights.shape[3] == width
    
        rpn_bbox_outside_weights = np.concatenate((rpn_bbox_outside_weights,bbox_outside_weights))
    
    return rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights


def anchor_target_layer_ws(rpn_cls_score, gt_boxes, num_gt_boxes, im_info, data, _feat_stride = [16,], anchor_scales = [4 ,8, 16, 32]):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    """This function is for cases with empty 'gt_boxes' (weakly supervised)"""
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]

    height, width = rpn_cls_score.shape[1:3]
    A = _num_anchors
    batch_size = rpn_cls_score.shape[0]

    rpn_labels = np.empty((batch_size, 1, A * height, width), dtype=np.float32)
    rpn_labels.fill(-1)
    rpn_bbox_targets = np.zeros((batch_size, A * 4, height, width), dtype=np.float32)
    rpn_bbox_inside_weights = np.zeros((batch_size, A * 4, height, width), dtype=np.float32)
    rpn_bbox_outside_weights = np.zeros((batch_size, A * 4, height, width), dtype=np.float32)
    
    return rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights


def anchor_target_layer_joint(rpn_cls_score, gt_boxes, num_gt_boxes, im_info, data, is_training, _feat_stride=[16,], anchor_scales=[4 ,8, 16, 32], dataset='SNUBH'):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
        
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]

    if DEBUG:
        print 'anchors:'
        print _anchors
        print 'anchor shapes:'
        print np.hstack(( 
            _anchors[:, 2::4] - _anchors[:, 0::4],
            _anchors[:, 3::4] - _anchors[:, 1::4],
        ))
        _counts = cfg.EPS
        _sums = np.zeros((1, 4))
        _squared_sums = np.zeros((1, 4))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0

    # allow boxes to sit over the edge by a small amount
    _allowed_border =  0

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]
    batch_size = rpn_cls_score.shape[0]
    
    #### same for all batch element ####
    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)
    #### same for all batch element ####    
    
    rpn_labels = np.zeros((0, 1, A * height, width), dtype=np.float32)
    rpn_bbox_targets = np.zeros((0, A * 4, height, width), dtype=np.float32)
    rpn_bbox_inside_weights = np.zeros((0, A * 4, height, width), dtype=np.float32)
    rpn_bbox_outside_weights = np.zeros((0, A * 4, height, width), dtype=np.float32)
    for i in xrange(cfg.TRAIN.IMS_PER_BATCH):
        
        t_im_info = im_info[i,:]
        t_rpn_cls_score = rpn_cls_score[[i],:,:,:]
        t_num_gt_boxes =  num_gt_boxes[i]
        t_gt_boxes = gt_boxes[i,:t_num_gt_boxes,:]
    
        assert t_rpn_cls_score.shape[0] == 1, \
            'Only single item batches are supported'

        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width
            print ''
            print 'im_size: ({}, {})'.format(t_im_info[0], t_im_info[1])
            print 'scale: {}'.format(t_im_info[2])
            print 'height, width: ({}, {})'.format(height, width)
            print 'rpn: gt_boxes.shape', t_gt_boxes.shape
            print 'rpn: gt_boxes', t_gt_boxes
    
        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -_allowed_border) &
            (all_anchors[:, 1] >= -_allowed_border) &
            (all_anchors[:, 2] < t_im_info[1] + _allowed_border) &  # width
            (all_anchors[:, 3] < t_im_info[0] + _allowed_border)    # height
        )[0]
    
        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', len(inds_inside)
    
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print 'anchors.shape', anchors.shape
    
        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)
        
        if dataset=='SNUBH':
            ################## SNUBH dataset ##################  
            # 'gt_boxes' contains boxes not only for fg but also for bg
            # while 'gt_boxes' in 'pascal voc' only has boxes for fg 
            b_pos = np.transpose(t_gt_boxes[:,4]!=0) ## added by syshin
            num_pos = np.sum(b_pos) ## added by syshin
            exist_neg = (t_gt_boxes.shape[0] != num_pos)
        
            # overlaps between the anchors and the gt boxes
            # overlaps (ex, gt)
            overlaps_pos = bbox_overlaps(
                    np.ascontiguousarray(anchors, dtype=np.float),
                    np.ascontiguousarray(t_gt_boxes[:num_pos,:], dtype=np.float))
                
            argmax_overlaps_pos = overlaps_pos.argmax(axis=1)
            max_overlaps_pos = overlaps_pos[np.arange(len(inds_inside)), argmax_overlaps_pos]
            gt_argmax_overlaps_pos = overlaps_pos.argmax(axis=0)
            gt_max_overlaps_pos = overlaps_pos[gt_argmax_overlaps_pos,
                                       np.arange(overlaps_pos.shape[1])]
            gt_argmax_overlaps_pos = np.where(overlaps_pos == gt_max_overlaps_pos)[0]
            
            if exist_neg:
                overlaps_neg = bbox_overlaps_ui(
                        np.ascontiguousarray(anchors, dtype=np.float),
                        np.ascontiguousarray(t_gt_boxes[num_pos:,:], dtype=np.float))
                
                argmax_overlaps_neg = overlaps_neg.argmax(axis=1)
                max_overlaps_neg = overlaps_neg[np.arange(len(inds_inside)), argmax_overlaps_neg]
               
            if not cfg.TRAIN.RPN_CLOBBER_POSITIVES and exist_neg:
                # assign bg labels first so that positive labels can clobber them
                labels[max_overlaps_neg >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 0
        
            # fg label: for each gt, anchor with highest overlap
            labels[gt_argmax_overlaps_pos] = 1
        
            # fg label: above threshold IOU
            labels[max_overlaps_pos >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
            ################## SNUBH dataset ##################  
        else:
            if dataset=='SNUBH_FG':
                ################## SNUBH dataset(w/o bg boxes) ##################
                # 'gt_boxes' contains boxes not only for fg but also for bg
                # but, here we use only the fg boxes
                b_pos = np.transpose(t_gt_boxes[:,4]!=0)
                num_pos = np.sum(b_pos)
                t_gt_boxes = t_gt_boxes[:num_pos,:]
                ################## SNUBH dataset(w/o bg boxes) ##################
            
            ################## UDIAT dataset ##################
            # 'gt_boxes' has boxes only for fg
            # no action is required
            ################## UDIAT dataset ##################
    
            # overlaps between the anchors and the gt boxes
            # overlaps (ex, gt)
            overlaps = bbox_overlaps(
                    np.ascontiguousarray(anchors, dtype=np.float),
                    np.ascontiguousarray(t_gt_boxes, dtype=np.float))
            
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                       np.arange(overlaps.shape[1])]
            gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
            
            if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels first so that positive labels can clobber them
                labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    
            # fg label: for each gt, anchor with highest overlap
            labels[gt_argmax_overlaps] = 1
        
            # fg label: above threshold IOU
            labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
        
            if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels last so that negative labels can clobber positives
                labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
    
        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1
    
        # subsample negative labels if we have too many
        # num_bg = batch_size - possible_num_fg
        # this may result in the class imbalance in batches
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            #print "was %s inds, disabling %s, now %s inds" % (
                #len(bg_inds), len(disable_inds), np.sum(labels == 0))
    
        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if dataset=='SNUBH':
            bbox_targets = _compute_targets(anchors, t_gt_boxes[argmax_overlaps_pos, :])    
        else:
            bbox_targets = _compute_targets(anchors, t_gt_boxes[argmax_overlaps, :])
    
        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)
    
        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights
    
        if DEBUG:
            _sums += bbox_targets[labels == 1, :].sum(axis=0)
            _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            _counts += np.sum(labels == 1)
            means = _sums / _counts
            stds = np.sqrt(_squared_sums / _counts - means ** 2)
            print 'means:'
            print means
            print 'stdevs:'
            print stds
    
        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)
    
        if DEBUG:
            print 'rpn: max max_overlap_pos', np.max(max_overlaps_pos)
            if exist_neg:
                print 'rpn: max max_overlap_neg', np.max(max_overlaps_neg)
            #print 'rpn: max max_overlap', np.max(max_overlaps)
            print 'rpn: num_positive', np.sum(labels == 1)
            print 'rpn: num_negative', np.sum(labels == 0)
            _fg_sum += np.sum(labels == 1)
            _bg_sum += np.sum(labels == 0)
            _count += 1
            print 'rpn: num_positive avg', _fg_sum / _count
            print 'rpn: num_negative avg', _bg_sum / _count
    
        # labels
        #pdb.set_trace()
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        rpn_labels = np.concatenate((rpn_labels,labels))
    
        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    
        rpn_bbox_targets = np.concatenate((rpn_bbox_targets,bbox_targets))
        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        #assert bbox_inside_weights.shape[2] == height
        #assert bbox_inside_weights.shape[3] == width
    
        rpn_bbox_inside_weights = np.concatenate((rpn_bbox_inside_weights,bbox_inside_weights))
    
        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        #assert bbox_outside_weights.shape[2] == height
        #assert bbox_outside_weights.shape[3] == width
    
        rpn_bbox_outside_weights = np.concatenate((rpn_bbox_outside_weights,bbox_outside_weights))

    if is_training:

        labels = np.empty((cfg.TRAIN.WS_IMS_PER_BATCH, 1, A * height, width), dtype=np.float32)
        labels.fill(-1)
        rpn_labels = np.concatenate((rpn_labels,labels))
        
        bbox_targets = np.zeros((cfg.TRAIN.WS_IMS_PER_BATCH, A * 4, height, width), dtype=np.float32)
        rpn_bbox_targets = np.concatenate((rpn_bbox_targets,bbox_targets))
        
        bbox_inside_weights = np.zeros((cfg.TRAIN.WS_IMS_PER_BATCH, A * 4, height, width), dtype=np.float32)
        rpn_bbox_inside_weights = np.concatenate((rpn_bbox_inside_weights,bbox_inside_weights))
            
        bbox_outside_weights = np.zeros((cfg.TRAIN.WS_IMS_PER_BATCH, A * 4, height, width), dtype=np.float32)
        rpn_bbox_outside_weights = np.concatenate((rpn_bbox_outside_weights,bbox_outside_weights)) 

    return rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights
    

def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
