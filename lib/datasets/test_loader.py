import numpy as np
import torch
import torchvision as tv
import torch.utils.data as data


class TestLoader(data.DataLoader):
    def __init__(self, roidb, batch_size=1, shuffle=False, num_workers=1, pin_memory=False):
        # save parameters as properties
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle

        # infer properties from roidb
        self.size = np.sum([x['frame_seg_len'] for x in self.roidb])
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        self.data_name = ['data', 'im_info', 'data_key', 'feat_key']
        self.label_name = None

        #
        self.cur_roidb_index = 0
        self.cur_frameid = 0
        self.data_key = None
        self.key_frameid = 0
        self.cur_seg_len = 0
        self.key_frame_flag = -1

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = []
        self.im_info = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    def __len__(self):
        return self.size

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            self.cur_frameid += 1
            if self.cur_frameid == self.cur_seg_len:
                self.cur_roidb_index += 1
                self.cur_frameid = 0
                self.key_frameid = 0
            elif self.cur_frameid - self.key_frameid == self.cfg.TEST.KEY_FRAME_INTERVAL:
                self.key_frameid = self.cur_frameid
            return self.im_info, self.key_frame_flag, mx.io.DataBatch(data=self.data, label=self.label,
                                                                      pad=self.getpad(), index=self.getindex(),
                                                                      provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_roidb = self.roidb[self.cur_roidb_index].copy()
        cur_roidb['image'] = cur_roidb['pattern'] % self.cur_frameid
        self.cur_seg_len = cur_roidb['frame_seg_len']
        data, label, im_info = get_rpn_testbatch([cur_roidb], self.cfg)
        if self.key_frameid == self.cur_frameid:  # key frame
            self.data_key = data[0]['data'].copy()
            if self.key_frameid == 0:
                self.key_frame_flag = 0
            else:
                self.key_frame_flag = 1
        else:
            self.key_frame_flag = 2
        extend_data = [{'data': data[0]['data'],
                        'im_info': data[0]['im_info'],
                        'data_key': self.data_key,
                        'feat_key': np.zeros((1, self.cfg.network.APN_FEAT_DIM, 1, 1))}]
        self.data = [[mx.nd.array(extend_data[i][name])
                      for name in self.data_name] for i in range(len(data))]
        self.im_info = im_info
