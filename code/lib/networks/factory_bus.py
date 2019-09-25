# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

# coded by syshin
# modified from the original code 'factory.py'

"""Factory method for easily getting a network by name."""

__sets = {}

import networks.VGGnet_train_bus
import networks.VGGnet_train_bus_alter
import networks.VGGnet_test_bus
import networks.Resnet_train_bus
import networks.Resnet_test_bus
import pdb
import tensorflow as tf

#__sets['VGGnet_train'] = networks.VGGnet_train()

#__sets['VGGnet_test'] = networks.VGGnet_test()


def get_network(name, net_depth, dataset, norm_type=None):
    """Get a network by name."""
    #if not __sets.has_key(name):
    #    raise KeyError('Unknown dataset: {}'.format(name))
    #return __sets[name]
    if name == 'VGGnet_train':
        return networks.VGGnet_train_bus(dataset)
    elif name == 'VGGnet_train_alter':
        return networks.VGGnet_train_bus_alter(dataset)
    elif name == 'VGGnet_test':
        return networks.VGGnet_test_bus()
    elif name == 'Resnet_train':
        return networks.Resnet_train_bus(net_depth, dataset, norm_type)    
    elif name == 'Resnet_test':
        return networks.Resnet_test_bus(net_depth, dataset, norm_type)
    else:
       raise KeyError('Unknown dataset: {}'.format(name))
    

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()