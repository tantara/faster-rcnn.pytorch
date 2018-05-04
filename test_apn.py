# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader, testBatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16, vgg16_key, vgg16_cur
from model.faster_rcnn.resnet import resnet, resnet_key, resnet_cur
from datasets.test_loader import TestLoader
from multiprocessing.pool import ThreadPool as Pool
# from torch.multiprocessing.pool import Pool
from core.tester import pred_eval
from easydict import EasyDict as edict

import pdb


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="/SSD/tantara/models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--ignore_cache', dest='ignore_cache',
                        help='ignore output cache detection files',
                        default=False, type=bool)
    parser.add_argument('--gpus', dest='gpus',
                        help='gpus',
                        default='0', type=str)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--quant_method', default='linear',
                        help='linear|minmax|log|tanh')

    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "imagenet":
        args.imdb_name = "imagenet_DET_train_30classes+VID_train_15frames"
        args.imdbval_name = "imagenet_VID_val_videos"
        args.imdbtest_name = "imagenet_VID_val_frames"
        args.set_cfgs = ['ANCHOR_SCALES',
                         '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    else:
        raise

    args.cfg_file = "cfgs/{}_ls.yml".format(
        args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if args.cuda:
        cfg.CUDA = True

    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    # im_data = torch.FloatTensor(1)
    # im_info = torch.FloatTensor(1)
    # num_boxes = torch.LongTensor(1)
    # gt_boxes = torch.FloatTensor(1)
    # feat = torch.FloatTensor(1)
    #
    # # ship to cuda
    # if args.cuda:
    #   im_data = im_data.cuda()
    #   im_info = im_info.cuda()
    #   num_boxes = num_boxes.cuda()
    #   gt_boxes = gt_boxes.cuda()
    #   feat = feat.cuda()
    #
    # # make variable
    # im_data = Variable(im_data, volatile=True)
    # im_info = Variable(im_info, volatile=True)
    # num_boxes = Variable(num_boxes, volatile=True)
    # gt_boxes = Variable(gt_boxes, volatile=True)
    # feat = Variable(feat, volatile=True)

    cfg.TRAIN.USE_FLIPPED = False
    cfg.network = edict()
    cfg.network.APN_FEAT_DIM = 1024  # FIXME
    # cfg.TEST = edict()
    cfg.TEST.KEY_FRAME_INTERVAL = 10  # FIXME
    cfg.class_agnostic = args.class_agnostic

    imdb, roidb, ratio_list, ratio_index = combined_roidb(
        args.imdbval_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))
    print('ratio_list', len(ratio_list), 'ratio_index', len(ratio_index))

    gpus = list(map(int, args.gpus.split(',')))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in gpus])
    print('Available devices ', torch.cuda.device_count())
    print('current device', torch.cuda.current_device())

    roidbs = [[] for x in gpus]
    roidbs_seg_lens = np.zeros(len(gpus), dtype=np.int)
    for x in roidb:
        gpu_id = np.argmin(roidbs_seg_lens)
        roidbs[gpu_id].append(x)
        roidbs_seg_lens[gpu_id] += x['frame_seg_len']

    print(roidbs_seg_lens)
    save_name = 'faster_rcnn_10'
    output_dir = get_output_dir(imdb, save_name)
    cfg.output_dir = output_dir
    datasets = [testBatchLoader(x, ratio_list, ratio_index, args.batch_size,
                                imdb.num_classes, normalize=False, cfg=cfg) for x in roidbs]
    # datasets = [TestLoader(x, ratio_list, ratio_index, args.batch_size, \
    #                       imdb.num_classes, training=False, normalize = False) for x in roidbs]
    dataloaders = [torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                               shuffle=False, num_workers=0,
                                               pin_memory=True) for dataset in datasets]

    class Predictor(object):
        def __init__(self, net, dataloader, cfg, imdb, gpu_id):
            self.net = net
            self.dataloader = dataloader
            self.gpu_id = gpu_id
            self.cfg = cfg
            self.imdb = imdb

        # def predict():
        #     num_images = len(self.dataloader.size)
        #     all_boxes = [[[] for _ in xrange(num_images)]
        #                  for _ in xrange(self.imdb.num_classes)]
        #
        #     data_iter = iter(dataloader)
        #
        #     _t = {'im_detect': time.time(), 'misc': time.time()}
        #     det_file = os.path.join(self.cfg.output_dir, 'detections_{0}.pkl'.format(self.gpu_id))
        #
        #     empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
        #
        #     if not args.ignore_cache and os.path.exists(det_file):
        #         print('load detections from', det_file)
        #         with open(det_file, 'rb') as f:
        #             all_boxes = pickle.load(f)
        #     else:
        #         print('[*] test_apn', 'num_images', num_images)
        #         all_boxes = []
        #
        #         with open(det_file, 'wb') as f:
        #             pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
        #
        #     print('Evaluating detections', len(all_boxes), 'boxes\n')
        #     return all_boxes

    def get_predictor(net, dataloader, cfg, imdb, gpu_id):
        # net = net.clone()
        net.share_memory()
        if cfg.CUDA:
            print('cuda', gpu_id)
            net = net.cuda(gpu_id)
        net.eval()

        predictor = Predictor(net, dataloader, cfg, imdb, gpu_id)
        return predictor

    # initilize the network here.
    def get_models(args, cfg):
        input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
        if not os.path.exists(input_dir):
            raise Exception(
                'There is no input directory for loading network from ' + input_dir)
        load_name = os.path.join(input_dir,
                                 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

        if args.net == 'vgg16':
            fasterRCNNKey = vgg16_key(
                imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
            fasterRCNNCur = vgg16_cur(
                imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res101':
            fasterRCNNKey = resnet_key(
                imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
            fasterRCNNCur = resnet_cur(
                imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res50':
            fasterRCNNKey = resnet_key(
                imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
            fasterRCNNCur = resnet_cur(
                imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
        elif args.net == 'res152':
            fasterRCNNKey = resnet_key(
                imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
            fasterRCNNCur = resnet_cur(
                imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()

        fasterRCNNKey.create_architecture()
        fasterRCNNCur.create_architecture()

        print("load checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        fasterRCNNKey.load_state_dict(checkpoint['model'])
        fasterRCNNCur.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print('load model successfully!')

        return [fasterRCNNKey, fasterRCNNCur]

    models = [get_models(args, cfg) for _ in gpus]

    # create predictor
    key_predictors = [get_predictor(
        models[i][0], dataloaders[i], cfg, imdb, gpu_id) for i, gpu_id in enumerate(gpus)]
    cur_predictors = [get_predictor(
        models[i][1], dataloaders[i], cfg, imdb, gpu_id) for i, gpu_id in enumerate(gpus)]

    def pred_eval_multiprocess(gpus, key_predictors, cur_predictors, test_datas, imdb, cfg, roidbs_seg_lens=roidbs_seg_lens, vis=False, thresh=1e-4, ignore_cache=True,):
        start = time.time()
        if len(gpus) == 1:
            res = [pred_eval(0, 0, key_predictors[0], cur_predictors[0], test_datas[0],
                             roidbs[0], imdb, cfg, roidbs_seg_lens, vis, thresh, ignore_cache), ]
        else:
            print('multiple processes')
            pool = Pool(processes=len(gpus))  # , ctx=None)
            multiple_results = [pool.apply_async(pred_eval, args=(i, gpu_id, key_predictors[i], cur_predictors[i], test_datas[i],
                                                                  roidbs[i], imdb, cfg, roidbs_seg_lens, vis, thresh, ignore_cache)) for i, gpu_id in enumerate(gpus)]
            pool.close()
            pool.join()
            res = [res.get() for res in multiple_results]
    # imdb.evaluate_detections(all_boxes, output_dir)
        info_str = imdb.evaluate_detections_multiprocess(res, cfg.output_dir)
        end = time.time()
        print("test time: %0.4fs" % (end - start))

    # start detection
    #pred_eval(0, key_predictors[0], cur_predictors[0], dataloaders[0], imdb, vis=vis, ignore_cache=ignore_cache, thresh=thresh)
    pred_eval_multiprocess(gpus, key_predictors, cur_predictors, dataloaders, imdb, cfg,
                           roidbs_seg_lens=roidbs_seg_lens, vis=vis, ignore_cache=args.ignore_cache, thresh=thresh)
