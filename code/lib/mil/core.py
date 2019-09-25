# functions for weakly supervised training
# coded by syshin

from fast_rcnn.config import cfg
import os
import tensorflow as tf
import numpy as np
import pdb


def get_bag_logit(instance_logits, batch_inds, num_classes, bag_labels, batch_size, funcs):
    """Get responses of each bag.
    Args:
        instance_logits: Logits tensor, float32 - [None, num_classes].
        batch_inds: A index tensor indicating what bag each instance is from, int32 - [None, 1]
        num_classes: Number of classes - []
        bag_labels: Bag(image-level) labels, int32 - [batch_size, 1]
        batch_size: Batch size, this is different for supervised and weakly supervised dataset - []
        funcs: functions to extract a bag logit from instance logits
    Returns:
        bag_logits: Logits tensor, float32 - [batch_size, num_classes].
    """
    
    num_boxes = tf.zeros([0,1], dtype=tf.int32)
    for i in xrange(batch_size):
        t_num_boxes = tf.reshape(tf.reduce_sum(tf.cast(tf.equal(batch_inds,i), dtype=tf.int32)), [1,1])
        num_boxes = tf.concat(axis=0, values=[num_boxes,t_num_boxes])
  
    bag_logits = tf.zeros([0, num_classes], dtype=tf.float32)
    scale_factors = tf.zeros([0,], dtype=tf.float32)
    cum_num_boxes = tf.constant([0], tf.int32, [1, 1])
    for i in xrange(batch_size):
        begin = tf.reshape(tf.concat(axis=1, values=[tf.reshape(cum_num_boxes, [1, 1]), tf.constant([0], tf.int32, [1, 1])]), [2]) 
        size = tf.reshape(tf.concat(axis=1, values=[tf.reshape(tf.slice(num_boxes, [i,0], [1,-1]), [1, 1]), 
            tf.constant([-1], tf.int32, [1,1])]), [2])
        cum_num_boxes = tf.add(cum_num_boxes, tf.reshape(tf.slice(num_boxes, [i,0], [1,-1]), [1, 1]))
    
        cur_logits = tf.slice(instance_logits, begin, size)
        cur_bag_label = tf.reshape(tf.slice(bag_labels, [i,0], [1,-1]), [])
        max_logit = tf.cond(tf.equal(cur_bag_label,1), \
                            lambda: funcs[0](cur_logits), \
                            lambda: funcs[1](cur_logits))
        bag_logits = tf.concat(axis=0, values=[bag_logits, max_logit])
        scale_factors = tf.concat(axis=0, values=[scale_factors, tf.reshape(tf.slice(tf.nn.softmax(max_logit), [0,cur_bag_label], [1,1]), [1,])])
    
    return bag_logits, scale_factors


def get_ben_max_logit(cur_logits):
    """Help function"""
    
    ben_logits = tf.slice(cur_logits, [0,1], [-1,1]) # hard coding!!
    idx_max_instance = tf.to_int32(tf.reshape(tf.arg_max(ben_logits,0), [1, 1]))
    sel_begin = tf.reshape(tf.concat(axis=1, values=[idx_max_instance, tf.constant([0], tf.int32, [1, 1])]), [2])
    max_logit = tf.slice(cur_logits, sel_begin, [1, -1])
    
    return max_logit
    

def get_mal_max_logit(cur_logits):
    """Help function"""
    
    mal_logits = tf.slice(cur_logits, [0,2], [-1,1]) # hard coding!!
    idx_max_instance = tf.to_int32(tf.reshape(tf.arg_max(mal_logits,0), [1, 1]))
    sel_begin = tf.reshape(tf.concat(axis=1, values=[idx_max_instance, tf.constant([0], tf.int32, [1, 1])]), [2])
    max_logit = tf.slice(cur_logits, sel_begin, [1, -1])
    
    return max_logit


def get_mean_ben_logit(cur_logits):
    """Help function"""

    return tf.concat(axis=1, values=[tf.constant([0], tf.float32, [1, 1]), tf.reshape(tf.reduce_mean(tf.slice(cur_logits, [0,1], [-1,1])), [1, 1]), tf.constant([0], tf.float32, [1, 1])])


def get_disc_max_logit(cur_logits):
    """Help function"""

    disc_logits = tf.reduce_max(tf.slice(cur_logits, [0,1], [-1,-1]), axis=1)
    idx_max_instance = tf.to_int32(tf.reshape(tf.arg_max(disc_logits,0), [1, 1]))
    sel_begin = tf.reshape(tf.concat(axis=1, values=[idx_max_instance, tf.constant([0], tf.int32, [1, 1])]), [2])
    max_logit = tf.slice(cur_logits, sel_begin, [1, -1])
    
    return max_logit
    

def get_mass_max_logit(cur_logits):
    """Help function"""

    bg_logits = tf.slice(cur_logits, [0,0], [-1,1]) # hard coding!!
    idx_min_instance = tf.to_int32(tf.reshape(tf.arg_min(bg_logits,0), [1, 1]))
    sel_begin = tf.reshape(tf.concat(axis=1, values=[idx_min_instance, tf.constant([0], tf.int32, [1, 1])]), [2])
    max_logit = tf.slice(cur_logits, sel_begin, [1, -1])
    
    return max_logit