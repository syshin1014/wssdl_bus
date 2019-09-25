# coded by syshin
# datalayer for joint training

from fast_rcnn.config import cfg
from roi_data_layer.minibatch_bus import get_minibatch_joint
import numpy as np

class RoIDataLayerJoint(object):
    """Fast R-CNN data layer used for training.
    We can simultaneously handle supervised and weakly supervised datasets in this layer."""

    def __init__(self, roidb_s, roidb_ws, net_name, num_classes, is_training):
        """Set the roidbs to be used by this layer during training."""
        self._roidb_s = roidb_s
        self._roidb_ws = roidb_ws
        self._net_name = net_name
        self._num_classes = num_classes
        self._is_training = is_training
        
        if self._is_training:
            self._shuffle_roidb_s_inds()
            self._shuffle_roidb_ws_inds()
            """self._roidb_s_inds()
            self._roidb_ws_inds()"""
        else:
            self._roidb_s_inds()
            self._roidb_ws_inds()

    def _shuffle_roidb_s_inds(self):
        """Randomly permute the training roidb."""
        self._perm_s = np.random.permutation(np.arange(len(self._roidb_s)))
        self._cur_s = 0
    
    def _shuffle_roidb_ws_inds(self):
        """Randomly permute the training roidb."""
        self._perm_ws = np.random.permutation(np.arange(len(self._roidb_ws)))
        self._cur_ws = 0
    
    def _roidb_s_inds(self):
        """Permute the training roidb."""
        self._perm_s = np.arange(len(self._roidb_s))
        self._cur_s = 0
        
    def _roidb_ws_inds(self):
        """Permute the training roidb."""
        self._perm_ws = np.arange(len(self._roidb_ws))
        self._cur_ws = 0        
        
    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        
        if cfg.TRAIN.HAS_RPN:
            if self._cur_s + cfg.TRAIN.IMS_PER_BATCH > len(self._roidb_s):
                if self._is_training:
                    self._shuffle_roidb_s_inds()
                    #self._roidb_s_inds()
                else:
                    self._roidb_s_inds()

            db_inds_s = self._perm_s[self._cur_s:self._cur_s + cfg.TRAIN.IMS_PER_BATCH]
            self._cur_s += cfg.TRAIN.IMS_PER_BATCH
            
            if self._cur_ws + cfg.TRAIN.WS_IMS_PER_BATCH > len(self._roidb_ws):
                if self._is_training:
                    self._shuffle_roidb_ws_inds()
                    #self._roidb_ws_inds()
                else:
                    self._roidb_ws_inds()

            db_inds_ws = self._perm_ws[self._cur_ws:self._cur_ws + cfg.TRAIN.WS_IMS_PER_BATCH ]
            self._cur_ws += cfg.TRAIN.WS_IMS_PER_BATCH 
        else:
            # sample images
            db_inds = np.zeros((cfg.TRAIN.IMS_PER_BATCH), dtype=np.int32)
            i = 0
            while (i < cfg.TRAIN.IMS_PER_BATCH):
                ind = self._perm_s[self._cur_s]
                num_objs = self._roidb_s[ind]['boxes'].shape[0]
                if num_objs != 0:
                    db_inds[i] = ind
                    i += 1

                self._cur_s += 1
                if self._cur_s >= len(self._roidb_s):
                    self._shuffle_roidb_s_inds()
                    #self._roidb_s_inds()

        return db_inds_s, db_inds_ws

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds_s,db_inds_ws = self._get_next_minibatch_inds()
        minibatch_db_s = [self._roidb_s[i] for i in db_inds_s]
        minibatch_db_ws = [self._roidb_ws[i] for i in db_inds_ws]
        return get_minibatch_joint(minibatch_db_s, minibatch_db_ws, self._net_name, self._num_classes, self._is_training)
            
    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs