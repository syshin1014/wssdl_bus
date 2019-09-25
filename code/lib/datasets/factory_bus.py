# coded by syshin
# modified from the script 'factory.py' by Ross Girshick


"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.bus
import numpy as np

# BUS dataset
for split in ['s_train', 's_train_10', 's_train_50', 's_train_100', 's_train_200', 's_train_400', 's_train_600', \
              'ws_train', 'ws_train_10', 'ws_train_50', 'ws_train_100', 'ws_train_200', 'ws_train_400', 'ws_train_600', \
              'train', 'reduced_ws_train', \
              'test', 'test_normal', \
              's_train_datasetB', 'test_datasetB']:
    name = 'bus_{}'.format(split)
    print name
    __sets[name] = (lambda split=split:
            datasets.bus(split))
 
def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()