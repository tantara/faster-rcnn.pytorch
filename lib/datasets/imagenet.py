from __future__ import print_function
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.imagenet
import os, sys
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import subprocess
import pdb
import pickle
from .imagenet_vid_eval import vid_eval

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class imagenet(imdb):
    def __init__(self, image_set, data_path):
        det_vid = image_set.split('_')[0]
        imdb.__init__(self, image_set)

        self._det_vid = det_vid
        self._image_set = image_set
        # self._devkit_path = devkit_path
        self._data_path = data_path
        # synsets_image = sio.loadmat(os.path.join(self._devkit_path, 'data', 'meta_det.mat'))
        # synsets_video = sio.loadmat(os.path.join(self._devkit_path, 'data', 'meta_vid.mat'))
        # self._classes_image = ('__background__',)
        # self._wnid_image = (0,)
        #
        # self._classes = ('__background__',)
        # self._wnid = (0,)

        # for i in xrange(200):
        #     self._classes_image = self._classes_image + (synsets_image['synsets'][0][i][2][0],)
        #     self._wnid_image = self._wnid_image + (synsets_image['synsets'][0][i][1][0],)
        #
        # for i in xrange(30):
        #     self._classes = self._classes + (synsets_video['synsets'][0][i][2][0],)
        #     self._wnid = self._wnid + (synsets_video['synsets'][0][i][1][0],)
        #
        # self._wnid_to_ind_image = dict(zip(self._wnid_image, xrange(201)))
        # self._class_to_ind_image = dict(zip(self._classes_image, xrange(201)))

        self._classes = ('__background__',  # always index 0
                        'airplane', 'antelope', 'bear', 'bicycle',
                        'bird', 'bus', 'car', 'cattle',
                        'dog', 'domestic_cat', 'elephant', 'fox',
                        'giant_panda', 'hamster', 'horse', 'lion',
                        'lizard', 'monkey', 'motorcycle', 'rabbit',
                        'red_panda', 'sheep', 'snake', 'squirrel',
                        'tiger', 'train', 'turtle', 'watercraft',
                        'whale', 'zebra')
        self._wnid = ('__background__',  # always index 0
                        'n02691156', 'n02419796', 'n02131653', 'n02834778',
                        'n01503061', 'n02924116', 'n02958343', 'n02402425',
                        'n02084071', 'n02121808', 'n02503517', 'n02118333',
                        'n02510455', 'n02342885', 'n02374451', 'n02129165',
                        'n01674464', 'n02484322', 'n03790512', 'n02324045',
                        'n02509815', 'n02411705', 'n01726692', 'n02355227',
                        'n02129604', 'n04468005', 'n01662784', 'n04530566',
                        'n02062744', 'n02391049')

        self._wnid_to_ind = dict(zip(self._wnid, xrange(31)))
        self._class_to_ind = dict(zip(self._classes, xrange(31)))

        #check for valid intersection between video and image classes
        # self._valid_image_flag = [0]*201

        # for i in range(1,201):
        #     if self._wnid_image[i] in self._wnid_to_ind:
        #         self._valid_image_flag[i] = 1

        self._image_ext = ['.JPEG']

        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        # assert os.path.exists(self._devkit_path), 'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # image_path = os.path.join(self._data_path, 'Data', self._image_set, index + self._image_ext[0])
        if self._det_vid == 'DET':
            image_path = os.path.join(self._data_path, 'Data', 'DET', index + '.JPEG')
        else:
            image_path = os.path.join(self._data_path, 'Data', 'VID', index + '.JPEG')

        # assert os.path.exists(image_path), 'path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self._data_path, 'ImageSets', self._image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            lines = [x.strip().split(' ') for x in f.readlines()]
        if len(lines[0]) == 2:
            image_set_index = [x[0] for x in lines]
            self.frame_id = [int(x[1]) for x in lines]
        else:
            image_set_index = ['%s/%06d' % (x[0], int(x[2])) for x in lines]
            self.pattern = [x[0]+'/%06d' for x in lines]
            self.frame_id = [int(x[1]) for x in lines]
            self.frame_seg_id = [int(x[2]) for x in lines]
            self.frame_seg_len = [int(x[3]) for x in lines]
        return image_set_index
        # return image_set_index, frame_id
        # """
        # Load the indexes listed in this dataset's image set file.
        # """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt

        # if self._image_set == 'train':
        #     image_set_file = os.path.join(self._data_path, 'ImageSets', 'trainr.txt')
        #     image_index = []
        #     if os.path.exists(image_set_file):
        #         f = open(image_set_file, 'r')
        #         data = f.read().split()
        #         for lines in data:
        #             if lines != '':
        #                 image_index.append(lines)
        #         f.close()
        #         return image_index
        #
        #     for i in range(1,200):
        #         print(i)
        #         image_set_file = os.path.join(self._data_path, 'ImageSets', 'DET', 'train_' + str(i) + '.txt')
        #         with open(image_set_file) as f:
        #             tmp_index = [x.strip() for x in f.readlines()]
        #             vtmp_index = []
        #             for line in tmp_index:
        #                 line = line.split(' ')
        #                 image_list = os.popen('ls ' + self._data_path + '/Data/DET/train/' + line[0] + '/*.JPEG').read().split()
        #                 tmp_list = []
        #                 for imgs in image_list:
        #                     tmp_list.append(imgs[:-5])
        #                 vtmp_index = vtmp_index + tmp_list
        #
        #         num_lines = len(vtmp_index)
        #         ids = np.random.permutation(num_lines)
        #         count = 0
        #         while count < 2000:
        #             image_index.append(vtmp_index[ids[count % num_lines]])
        #             count = count + 1
        #
        #     for i in range(1,201):
        #         if self._valid_image_flag[i] == 1:
        #             image_set_file = os.path.join(self._data_path, 'ImageSets', 'train_pos_' + str(i) + '.txt')
        #             with open(image_set_file) as f:
        #                 tmp_index = [x.strip() for x in f.readlines()]
        #             num_lines = len(tmp_index)
        #             ids = np.random.permutation(num_lines)
        #             count = 0
        #             while count < 2000:
        #                 image_index.append(tmp_index[ids[count % num_lines]])
        #                 count = count + 1
        #     image_set_file = os.path.join(self._data_path, 'ImageSets', 'trainr.txt')
        #     f = open(image_set_file, 'w')
        #     for lines in image_index:
        #         f.write(lines + '\n')
        #     f.close()
        # else:
        #     image_set_file = os.path.join(self._data_path, 'ImageSets', 'val.txt')
        #     with open(image_set_file) as f:
        #         image_index = [x.strip() for x in f.readlines()]
        # return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_imagenet_annotation(index) for index in range(0, len(self.image_index))]
        # gt_roidb = [self.load_vid_annotation(index) for index in range(0, len(self.image_set_index))]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb


    def _load_imagenet_annotation(self, iindex):
        """
        Load image and bounding boxes info from txt files of imagenet.
        """
        index = self.image_index[iindex]

        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)
        roi_rec['frame_id'] = self.frame_id[iindex]
        if hasattr(self,'frame_seg_id'):
            roi_rec['pattern'] = self.image_path_from_index(self.pattern[iindex])
            roi_rec['frame_seg_id'] = self.frame_seg_id[iindex]
            roi_rec['frame_seg_len'] = self.frame_seg_len[iindex]

        if self._det_vid == 'DET':
            filename = os.path.join(self._data_path, 'Annotations', 'DET', index + '.xml')
        else:
            filename = os.path.join(self._data_path, 'Annotations', 'VID', index + '.xml')

        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        size = data.getElementsByTagName('size')[0]
        roi_rec['height'] = float(get_data_from_tag(size, 'height'))
        roi_rec['width'] = float(get_data_from_tag(size, 'width'))

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.getElementsByTagName('bndbox')[0]
            # Make pixel indexes 0-based
            x1 = np.maximum(float(get_data_from_tag(obj, 'xmin')), 0)
            y1 = np.maximum(float(get_data_from_tag(obj, 'ymin')), 0)
            x2 = np.minimum(float(get_data_from_tag(obj, 'xmax')), roi_rec['width']-1)
            y2 = np.minimum(float(get_data_from_tag(obj, 'ymax')), roi_rec['height']-1)
            wnid = str(get_data_from_tag(obj, "name")).lower().strip()
            if not wnid in self._wnid_to_ind:
                continue
            cls = self._wnid_to_ind[wnid]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        roi_rec.update({'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'max_classes': overlaps.argmax(axis=1),
                'max_overlaps': overlaps.max(axis=1),
                'flipped' : False})
        return roi_rec

    def _get_result_file_template(self):
        """
        :return: a string template
        """
        res_file_folder = os.path.join(self.result_path, 'results')
        filename = 'det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path

    def _write_vid_results(self, all_boxes):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        print('Writing {} ImageNetVID results file'.format('all'))
        filename = self._get_result_file_template().format('all')
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self._image_index):
                for cls_ind, cls in enumerate(self._classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    # the imagenet expects 0-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:d} {:d} {:.4f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.
                                format(self.frame_id[im_ind], cls_ind, dets[k, -1],
                                       dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3]))

    def _write_vid_results_multiprocess(self, detections):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        print('Writing {} ImageNetVID results file'.format('all'))
        filename = self._get_result_file_template().format('all')
        print('eval', filename)
        with open(filename, 'wt') as f:
            for detection in detections:
                # print('detection', len(detection))
                all_boxes = detection[0]
                frame_ids = detection[1]
                print('all_boxes', len(all_boxes))
                print('frame_ids', len(frame_ids))
                for im_ind in range(len(all_boxes[0])):
                    for cls_ind, cls in enumerate(self._classes):
                        if cls == '__background__':
                            continue
                        dets = all_boxes[cls_ind][im_ind]
                        if len(dets) == 0:
                            continue
                        # the imagenet expects 0-based indices
                        for k in range(dets.shape[0]):
                            f.write('{:d} {:d} {:.4f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.
                                    format(frame_ids[im_ind], cls_ind, dets[k, -1],
                                           dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3]))

    def evaluate_detections(self, all_boxes, output_dir):
        # self._write_voc_results_file(all_boxes)
        self.result_path = output_dir
        result_dir = os.path.join(self.result_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        self._write_vid_results(all_boxes)
        info = self._do_python_eval(output_dir)

    def evaluate_detections_multiprocess(self, detections, output_dir):
        # self._write_voc_results_file(all_boxes)
        self.result_path = output_dir
        result_dir = os.path.join(self.result_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        print('detections', len(detections))
        self._write_vid_results_multiprocess(detections)
        info = self._do_python_eval_gen(output_dir)

    def _do_python_eval(self, output_dir):
        """
        python evaluation wrapper
        :return: info_str
        """
        info_str = ''
        annopath = os.path.join(self._data_path, 'Annotations', '{0!s}.xml')
        imageset_file = os.path.join(self._data_path, 'ImageSets', self._image_set + '.txt')
        annocache = os.path.join(self.cache_path, self.name + '_annotations.pkl')

        filename = self._get_result_file_template().format('all')
        ap = vid_eval(filename, annopath, imageset_file, self._wnid, annocache, ovthresh=0.5)
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            print('AP for {} = {:.4f}'.format(cls, ap[cls_ind-1]))
            info_str += 'AP for {} = {:.4f}\n'.format(cls, ap[cls_ind-1])
        print('Mean AP@0.5 = {:.4f}'.format(np.mean(ap)))
        info_str += 'Mean AP@0.5 = {:.4f}\n\n'.format(np.mean(ap))
        return info_str

    def _do_python_eval_gen(self, output_dir):
        """
        python evaluation wrapper
        :return: info_str
        """
        info_str = ''
        annopath = os.path.join(self._data_path, 'Annotations', '{0!s}.xml')
        imageset_file = os.path.join(self._data_path, 'ImageSets', self._image_set + '_eval.txt')
        annocache = os.path.join(self.cache_path, self.name + '_annotations.pkl')

        with open(imageset_file, 'w') as f:
            for i in range(len(self.pattern)):
                for j in range(self.frame_seg_len[i]):
                    f.write((self.pattern[i] % (self.frame_seg_id[i] + j)) + ' ' + str(self.frame_id[i] + j) + '\n')

        filename = self._get_result_file_template().format('all')
        ap = vid_eval(filename, annopath, imageset_file, self._wnid, annocache, ovthresh=0.5)
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            print('AP for {} = {:.4f}'.format(cls, ap[cls_ind-1]))
            info_str += 'AP for {} = {:.4f}\n'.format(cls, ap[cls_ind-1])
        print('Mean AP@0.5 = {:.4f}'.format(np.mean(ap)))
        info_str += 'Mean AP@0.5 = {:.4f}\n\n'.format(np.mean(ap))
        return info_str

if __name__ == '__main__':
    d = datasets.imagenet('val', '')
    res = d.roidb
    from IPython import embed; embed()
