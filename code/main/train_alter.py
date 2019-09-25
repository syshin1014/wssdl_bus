# Train the Faster-RCNN
# using alternating mini-batches each from supervised and weakly supervised sets.
# This script can be used also for supervised training by setting
# 'ws_start_iter' to be larger than 'max_iters',
# which means there is no weakly supervised iteration.
# coded by syshin

import _init_paths
from fast_rcnn.train_bus import get_training_roidb, train_net_alter
from fast_rcnn.test_bus import get_test_roidb # added by syshin
from fast_rcnn.config import cfg,cfg_from_file, cfg_from_list, get_output_dir, get_direct_output_dir
from datasets.factory_bus import get_imdb
from networks.factory_bus import get_network
import argparse
import pprint
import numpy as np
import sys
import pdb
import skimage.io

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--max_iters', help='maximum iterations', default=60000, type=int)
    parser.add_argument('--s_start_iter', help='start iter. for supervised learning', default=0, type=int)
    parser.add_argument('--s_end_iter', help='end iter. for supervised learning', default=60000, type=int)
    parser.add_argument('--ws_start_iter', help='start iter. for weakly supervised learning', default=0, type=int)
    parser.add_argument('--ws_end_iter', help='end iter. for weakly supervised learning', default=60000, type=int)
    parser.add_argument('--pretrained_model', help='path for a pretrained model', \
                        default='../pretrained_model/VGG_imagenet.npy', type=str)
    parser.add_argument('--set_cfgs', help='set config keys', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--randomize', help='randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument('--network', help='name of the network', default='VGGnet_train_alter', type=str)
    parser.add_argument('--net_depth', help='network depth', default=18, type=int) # speciifcally for Resnet (18, 34, 50, 101)
    parser.add_argument('--dataset', help='dataset', default='SNUBH', type=str)
    parser.add_argument('--norm_type', help='normalization type', default='BN', type=str) # [BN, GN]
    parser.add_argument('--opt', help='optimizer', default='adam', type=str) # [adam, amsgrad, sgd]
    parser.add_argument('--lr', help='initial learning rate', default=5e-04, type=float)
    parser.add_argument('--lr_scheduling', help='how to change the learning rate during training', \
                        default='const', type=str) # [const, pc, rop]
    parser.add_argument('--imdb_train_s', help='supervised training set', default='bus_test', type=str) # bus_s_train
    parser.add_argument('--imdb_train_ws', help='weakly supervised training set', default='bus_test', type=str) # bus_ws_train
    parser.add_argument('--imdb_test', help='test set', default='bus_test', type=str)
    parser.add_argument('--output_dir', help='output path', default='../trained_model/vgg_overfit', type=str)
    parser.add_argument('--qual_res_off', help='turn off saving qualitative results', \
                        default=True, action='store_false')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        
    imdb_train_s = get_imdb(args.imdb_train_s)
    print 'Loaded dataset `{:s}` for training'.format(imdb_train_s.name)
    roidb_train_s = get_training_roidb(imdb_train_s)
    
    imdb_train_ws = get_imdb(args.imdb_train_ws)
    print 'Loaded dataset `{:s}` for training'.format(imdb_train_ws.name)
    roidb_train_ws = get_training_roidb(imdb_train_ws)
    
    imdb_test = get_imdb(args.imdb_test)
    print 'Loaded dataset `{:s}` for test'.format(imdb_test.name)
    roidb_test = get_test_roidb(imdb_test)

    print 'Output will be saved to `{:s}`'.format(args.output_dir)

    network = get_network(args.network, args.net_depth, args.dataset, args.norm_type)
    print 'Use network `{:s}` in training'.format(args.network)

    train_net_alter(network, imdb_train_s, roidb_train_s, imdb_train_ws, roidb_train_ws, imdb_test, roidb_test, args.output_dir,
                    pretrained_model=args.pretrained_model,
                    max_iters=args.max_iters, s_start_iter=args.s_start_iter, s_end_iter=args.s_end_iter, 
                    ws_start_iter=args.ws_start_iter, ws_end_iter=args.ws_end_iter,
                    opt=args.opt, lr=args.lr, lr_scheduling=args.lr_scheduling, vis=args.qual_res_off)