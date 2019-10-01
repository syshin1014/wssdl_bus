# Test the Faster-RCNN
# coded by syshin

#import matplotlib
#matplotlib.use('Agg')

import _init_paths
from fast_rcnn.test_bus import test_net, get_test_roidb
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory_bus import get_imdb
from networks.factory_bus import get_network
import argparse
import pprint
import os
import tensorflow as tf


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
    parser.add_argument('--network', help='name of the network', \
                        default='VGGnet_test', type=str) # refer to the function 'get_network'
    parser.add_argument('--net_depth', help='network depth (only for ResNet [18, 34, 50, 101])', \
                        default=101, type=int)
    parser.add_argument('--norm_type', help='normalization type (only for ResNet)', default='BN', type=str)
    parser.add_argument('--trained_model', help='path for a trained model', \
                        default='../trained_model/vgg_800_4224/VGGnet_faster_rcnn.ckpt', type=str)
    parser.add_argument('--dataset', help='dataset', default='SNUBH', type=str)
    parser.add_argument('--imdb_test', help='test set', default='bus_test', type=str)
    parser.add_argument('--comp_mode', help='competition mode', action='store_true')
    parser.add_argument('--qual_res_off', help='turn off saving qualitative results', \
                        default=True, action='store_false')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()

    print('Called with args:')
    print(args)

    print('Using config:')
    pprint.pprint(cfg)

    imdb = get_imdb(args.imdb_test)
    imdb.competition_mode(args.comp_mode)
    roidb = get_test_roidb(imdb)

    network = get_network(args.network, args.net_depth, args.dataset, args.norm_type)
    print 'Use network `{:s}` in test'.format(args.network)
    
    output_dir = os.path.join(os.path.split(args.trained_model)[0], args.imdb_test)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # start a session
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    saver.restore(sess, args.trained_model)
    print ('Loading model weights from {:s}').format(args.trained_model) 

    test_net(sess, network, imdb, roidb, output_dir, vis=args.qual_res_off)
