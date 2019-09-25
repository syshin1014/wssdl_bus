# modified by syshin
# the original code is 'layer.py'

from fast_rcnn.config import cfg
#from roi_data_layer.minibatch import get_minibatch
from roi_data_layer.minibatch_bus import get_minibatch
import numpy as np

class RoIDataLayer(object):
    """Fast R-CNN data layer used for training."""

    def __init__(self, roidb, net_name, num_classes, is_training, is_ws):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._net_name = net_name
        self._num_classes = num_classes
        self._is_training = is_training
        self._is_ws = is_ws
        self._ims_per_batch = cfg.TRAIN.WS_IMS_PER_BATCH if is_ws else cfg.TRAIN.IMS_PER_BATCH
        
        if self._is_training:
            self._shuffle_roidb_inds()
        else:
            self._roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0
        
    def _roidb_inds(self):
        """Permute the training roidb."""
        self._perm = np.arange(len(self._roidb))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        
        if cfg.TRAIN.HAS_RPN:
            if self._cur + self._ims_per_batch > len(self._roidb):
                if self._is_training:
                    self._shuffle_roidb_inds()
                else:
                    self._roidb_inds()

            db_inds = self._perm[self._cur:self._cur + self._ims_per_batch]
            self._cur += self._ims_per_batch
        else:
            # sample images
            db_inds = np.zeros((self._ims_per_batch), dtype=np.int32)
            i = 0
            while (i < self._ims_per_batch):
                ind = self._perm[self._cur]
                num_objs = self._roidb[ind]['boxes'].shape[0]
                if num_objs != 0:
                    db_inds[i] = ind
                    i += 1

                self._cur += 1
                if self._cur >= len(self._roidb):
                    self._shuffle_roidb_inds()

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._net_name, self._num_classes, self._is_training, self._is_ws)
            
    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs