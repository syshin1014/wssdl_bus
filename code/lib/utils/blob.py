# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# modified by syshin

"""Blob helper functions."""

import numpy as np
#import cv2
#from PIL import Image
import skimage.transform
from fast_rcnn.config import cfg # added by syshin


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob

def prep_im_for_blob(im, net_name, pixel_means, pixel_stds, target_size, max_size, is_training, is_ws=False):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)/255.
    
    # added by syshin
    if is_ws:
        if cfg.TRAIN.USE_ROTATION:
            im = skimage.transform.rotate(im, np.random.uniform(-cfg.TRAIN.ROTATION_MAX_ANGLE,cfg.TRAIN.ROTATION_MAX_ANGLE), cval=pixel_means[0][0][0]/255.)
                                
        if cfg.TRAIN.USE_CROPPING:
            offsets_u = np.random.random_integers(0,cfg.TRAIN.CROPPING_MAX_MARGIN*im.shape[0])
            offsets_d = np.random.random_integers(1,cfg.TRAIN.CROPPING_MAX_MARGIN*im.shape[0])
            offsets_l = np.random.random_integers(0,cfg.TRAIN.CROPPING_MAX_MARGIN*im.shape[1])
            offsets_r = np.random.random_integers(1,cfg.TRAIN.CROPPING_MAX_MARGIN*im.shape[1])
            im = im[offsets_u:-offsets_d,offsets_l:-offsets_r,:]
        
    if is_training:
        if cfg.TRAIN.USE_BRIGHTNESS_ADJUSTMENT:
            im += np.random.uniform(-cfg.TRAIN.BRIGHTNESS_ADJUSTMENT_MAX_DELTA,cfg.TRAIN.BRIGHTNESS_ADJUSTMENT_MAX_DELTA)
            im = np.clip(im, 0, 1)
    
        if cfg.TRAIN.USE_CONTRAST_ADJUSTMENT:
            mm = np.mean(im)
            im = (im-mm)*np.random.uniform(cfg.TRAIN.CONTRAST_ADJUSTMENT_LOWER_FACTOR,cfg.TRAIN.CONTRAST_ADJUSTMENT_UPPER_FACTOR) + mm                         
            im = np.clip(im, 0, 1)       
    # added by syshin

    im -= pixel_means/255.
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    """im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)"""
    """im_pil = Image.fromarray(im)
    temp = im_pil.resize((np.round(im_shape[1]*im_scale),np.round(im_shape[0]*im_scale)), resample=Image.BILINEAR)
    im = np.array(temp)"""
    if net_name[:6]=='Resnet':
        im = skimage.transform.resize(im, [np.round(im_shape[0]*im_scale),np.round(im_shape[1]*im_scale)])/(pixel_stds/255.)
    elif net_name[:6]=='VGGnet':
        im = skimage.transform.resize(im, [np.round(im_shape[0]*im_scale),np.round(im_shape[1]*im_scale)])*255.
    
    return im, im_scale