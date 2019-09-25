# coded by syshin
# modified from the script 'pascal_voc.py' by Ross Girshick

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval_bus import voc_eval_bus
from fast_rcnn.config import cfg
import pdb


class bus(imdb):
    def __init__(self, image_set, bus_path=None):
        imdb.__init__(self, 'bus_' + image_set)
        self._image_set = image_set
        self._bus_path = self._get_default_path() if bus_path is None \
                            else bus_path
        #self._data_path = os.path.join(self._bus_path, 'BUS/aggr')
        self._data_path = self._bus_path
        self._classes = ('__background__', # always index 0
                         'benign', 'malignant')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.tif'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        #self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # BUS specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._bus_path), \
                'Path does not exist: {}'.format(self._bus_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'TIFFImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where BUS db is expected to be installed.
        """
        #return os.path.join(cfg.DATA_DIR, 'SNUBH_BUS')
        return cfg.DATA_DIR

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_bus_bbox_annotation(index)
                    for index in self._image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_bus_bbox_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)
        
        birads_diag = int(tree.findall('BIRADS')[0].find('diag').text)+1 # added by syshin

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas,
                'birads_diag' : birads_diag}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        """path = os.path.join(
            self._bus_path,
            'results',
            'BUS',
            'Main',
            filename)"""
        path = os.path.join(
            self._bus_path,
            'results',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output', suffix=None):
        """annopath = os.path.join(
            self._bus_path,
            'BUS',
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._bus_path,
            'BUS',
            'ImageSets',
            'Main',
            self._image_set + '.txt')"""
        annopath = os.path.join(
            self._bus_path,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._bus_path,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._bus_path, 'annotations_cache')
        aps = []
        nis = []
        noks = []
        corloc_list = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True
        #use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        #f_log = open(os.path.join(os.path.dirname(output_dir), 'batch_eval.txt'), 'a') # added by syshin
        f_log = open(os.path.join(output_dir, 'eval.txt'), 'w') # added by syshin
        """if suffix is not None:         
            f_log.write(output_dir+'_'+suffix+'\n')
        else:
            f_log.write(output_dir+'\n')
        f_log.flush()"""
        
        if 'bus_test_normal' in output_dir:
            ### just count the number of FPs ###
            num_fp_per_img_list = []
            for i, cls in enumerate(self._classes):
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                _, _, _, _, _, _, num_all_fps, num_fp_per_img = voc_eval_bus(
                    filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                    use_07_metric=use_07_metric, score_thresh=0.5)
                num_fp_per_img_list.append(num_fp_per_img)
                
            num_fp_per_img = np.array(num_fp_per_img_list)
            num_fp_per_img = np.sum(num_fp_per_img, axis=0)
                    
            f_log.write('Number of all FPs = {:d}'.format(np.sum(num_fp_per_img))+'\n')
            f_log.flush()    
            f_log.close()
            #np.save(os.path.join(os.path.dirname(output_dir),'num_fp_per_img'), num_fp_per_img)
            np.save(os.path.join(output_dir,'num_fp_per_img'), num_fp_per_img)
        else:
            all_arr_ok = np.zeros((0,))
            for i, cls in enumerate(self._classes):
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                rec, prec, ap, ni, nok, arr_ok, _, _ = voc_eval_bus(
                    filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                    use_07_metric=use_07_metric, score_thresh=0.5)
                aps += [ap]
                nis += [ni]
                noks += [nok]
                print('AP for {} = {:.4f}'.format(cls, ap))
                print('CorLoc for {} = {:.4f}'.format(cls, np.float(nok)/ni))
                """with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                    cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)"""
                f_log.write('AP for {} = {:.4f}'.format(cls, ap)+'\n') # deactivate if you are not interested
                f_log.write('CorLoc for {} = {:.4f}'.format(cls, np.float(nok)/ni)+'\n')
                f_log.flush()
                corloc_list.append(np.float(nok)/ni)
                all_arr_ok = np.concatenate((all_arr_ok,arr_ok))
                
            print('Mean AP = {:.4f}'.format(np.mean(aps)))
            print('CorLoc = {:.4f}'.format(np.float(np.sum(noks))/np.sum(nis)))
            f_log.write('Mean AP = {:.4f}'.format(np.mean(aps))+'\n') # deactivate if you are not interested
            f_log.write('CorLoc = {:.4f}'.format(np.float(np.sum(noks))/np.sum(nis))+'\n')
            corloc_list.append(np.float(np.sum(noks))/np.sum(nis))
            f_log.flush()
            #f_log.close()
            print('~~~~~~~~')
            print('Results:')
            for ap in aps:
                print('{:.3f}'.format(ap))
            print('{:.3f}'.format(np.mean(aps)))
            print('~~~~~~~~')
            print('')
            print('--------------------------------------------------------------')
            print('Results computed with the **unofficial** Python eval code.')
            print('Results should be very close to the official MATLAB eval code.')
            print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
            print('-- Thanks, The Management')
            print('--------------------------------------------------------------')
            
            ### FROC curves ###
            print('Drawing a FROC curve')
            curve_pts = [[],[],[]] # [all, benign, malignant]
            arr_score_thresh = np.arange(1.0, -0.01, -0.05)
            for cur_score_thresh in arr_score_thresh:
                for i, cls in enumerate(self._classes):
                    if cls == '__background__':
                        continue
                    filename = self._get_voc_results_file_template().format(cls)
                    rec, prec, ap, ni, nok, _, num_all_fps, num_fp_per_img = voc_eval_bus(
                        filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                        use_07_metric=use_07_metric, score_thresh=cur_score_thresh)
                    curve_pts[i].append((np.float(num_all_fps)/ni,np.float(nok)/ni)) # (x,y)
    
            for i in xrange(len(curve_pts[1])):
                curve_pts[0].append(((curve_pts[1][i][0]+curve_pts[2][i][0])/2, (curve_pts[1][i][1]+curve_pts[2][i][1])/2))
                    
            f_log.write(str(curve_pts))
            f_log.flush()    
            f_log.close()
            #np.save(os.path.join(os.path.dirname(output_dir),'froc_curve_pts'), curve_pts)
            np.save(os.path.join(output_dir,'froc_curve_pts'), curve_pts)
            #np.save(os.path.join(os.path.dirname(output_dir),'all_arr_ok'), all_arr_ok)
            np.save(os.path.join(output_dir,'all_arr_ok'), all_arr_ok)
            ### FROC curves ###
        
        return corloc_list

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._bus_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir, suffix=None):
        self._write_voc_results_file(all_boxes)
        corloc_list = self._do_python_eval(output_dir, suffix)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)
        return corloc_list

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.bus import bus
    d = bus('s')
    res = d.roidb
    from IPython import embed; embed()