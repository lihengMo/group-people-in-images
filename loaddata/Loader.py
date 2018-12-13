import mxnet as mx
import numpy as np

from loaddata.image import imdecode, resize, get_image, tensor_vstack, get_test_image

def load_test(im, roi_rec, roi_head, gt_index,mean,std):

    pair_im_1_tensor, pair_im_2_tensor, location_map_1_tensor, location_map_2_tensor, \
    gt_box_1, gt_box_2,r_map_tensor,headsize_1, headsize_2 = get_test_image(im,roi_rec,roi_head, gt_index,mean,std)
    grpid_1, subid_1 = gt_box_1[4:6]
    grpid_2, subid_2 = gt_box_2[4:6]

    # print(str(int(grpid_1))+'_'+str(int(subid_1)))
    # print(str(int(grpid_2))+'_'+str(int(subid_2)))

    if grpid_1 == grpid_2 and subid_1 == subid_2:
        label = 1
    else:
        label = 0

    pair_im_1_tensor = mx.nd.array(pair_im_1_tensor).expand_dims(0)
    pair_im_2_tensor = mx.nd.array(pair_im_2_tensor).expand_dims(0)
    location_map_1_tensor = mx.nd.array(location_map_1_tensor).expand_dims(0)
    location_map_2_tensor = mx.nd.array(location_map_2_tensor).expand_dims(0)
    r_map_tensor = mx.nd.array(r_map_tensor).expand_dims(0)
    headsize_1 = mx.nd.array(headsize_1).expand_dims(0)
    headsize_2 = mx.nd.array(headsize_2).expand_dims(0)
    return pair_im_1_tensor, pair_im_2_tensor, location_map_1_tensor, location_map_2_tensor, r_map_tensor, headsize_1,headsize_2,label


def generate_batch(pair_im_1_tensor, pair_im_2_tensor, location_map_1_tensor, location_map_2_tensor,r_map_tensor, headsize_1,headsize_2):
    """return batch"""
    data = [pair_im_1_tensor, pair_im_2_tensor, location_map_1_tensor, location_map_2_tensor,r_map_tensor, headsize_1,headsize_2]
    data_shapes = [('pair_im_1', pair_im_1_tensor.shape), ('pair_im_2', pair_im_2_tensor.shape),
                   ('loca_map_1', location_map_1_tensor.shape),  ('loca_map_2', location_map_2_tensor.shape),
                   ('r_map',r_map_tensor.shape), ('headsize_1',headsize_1.shape), ('headsize_2',headsize_2.shape)]
    data_batch = mx.io.DataBatch(data=data, label=None, provide_data=data_shapes, provide_label=None)
    return data_batch


class Loader(mx.io.DataIter):
    def __init__(self, roidb, roihead, batch_size, mean, std):
        super(Loader, self).__init__()

        # save parameters as properties
        self._roidb = roidb
        self._batch_size = batch_size
        self._mean = mean
        self._std = std
        self._roihead = roihead

        # infer properties from roidb
        self._size = len(self._roidb)
        self._index = np.arange(self._size)

        # decide data and label names (only for training)
        self._data_name = ['pair_im_1', 'pair_im_2','loca_map_1', 'loca_map_2', 'r_map','headsize_1','headsize_2']
        self._label_name = ['label']

        # status variable
        self._cur = 0
        self._data = None
        self._label = None
        self._fulldata = None

        # get first batch to fill in provide_data and provide_label
        self.next()
        self.reset()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self._data_name, self._data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self._label_name, self._label)]

    def reset(self):
        self._cur = 0

    def iter_next(self):
        return self._cur + self._batch_size <= self._size

    def next(self):
        if self.iter_next():
            data_batch = mx.io.DataBatch(data=self.getdata(), label=self.getlabel(),
                                         pad=self.getpad(), index=self.getindex(),
                                         provide_data=self.provide_data, provide_label=self.provide_label)
            self._cur += self._batch_size
            return data_batch
        else:
            raise StopIteration

    def getdata(self):
        indices = self.getindex()

        pair_im_1_tensor, pair_im_2_tensor, loca_map_1_tensor, loca_map_2_tensor, \
        gt_box_1, gt_box_2, r_map_tensor, headsize_1, headsize_2 = [], [], [], [], [], [], [], [], []
        for index in indices:
            roi_rec = self._roidb[index]
            roi_head = self._roihead[index]
            b_pair_im_1_tensor, b_pair_im_2_tensor, b_loca_map_1_tensor, b_loca_map_2_tensor, \
            b_gt_box_1, b_gt_box_2,b_r_map_tensor, b_headsize1, b_headsize2 = get_image(
                roi_rec, roi_head, self._mean, self._std)
            pair_im_1_tensor.append(b_pair_im_1_tensor)
            pair_im_2_tensor.append(b_pair_im_2_tensor)
            loca_map_1_tensor.append(b_loca_map_1_tensor)
            loca_map_2_tensor.append(b_loca_map_2_tensor)
            gt_box_1.append(b_gt_box_1)
            gt_box_2.append(b_gt_box_2)
            r_map_tensor.append(b_r_map_tensor)
            headsize_1.append(b_headsize1)
            headsize_2.append(b_headsize2)

        pair_im_1_tensor = mx.nd.array(tensor_vstack(pair_im_1_tensor, pad=0))
        pair_im_2_tensor = mx.nd.array(tensor_vstack(pair_im_2_tensor, pad=0))
        loca_map_1_tensor = mx.nd.array(tensor_vstack(loca_map_1_tensor, pad=0))
        loca_map_2_tensor = mx.nd.array(tensor_vstack(loca_map_2_tensor, pad=0))
        r_map_tensor = mx.nd.array(tensor_vstack(r_map_tensor, pad=0))
        gt_box_1 = mx.nd.array(tensor_vstack(gt_box_1, pad=-1))
        gt_box_2 = mx.nd.array(tensor_vstack(gt_box_2, pad=-1))
        headsize_1 = mx.nd.array(tensor_vstack(headsize_1, pad=-1))
        headsize_2 = mx.nd.array(tensor_vstack(headsize_2, pad=-1))

        self._fulldata = pair_im_1_tensor, pair_im_2_tensor, loca_map_1_tensor, loca_map_2_tensor, \
                         gt_box_1, gt_box_2, r_map_tensor, headsize_1,headsize_2
        self._data = pair_im_1_tensor, pair_im_2_tensor,loca_map_1_tensor,loca_map_2_tensor,\
                     r_map_tensor, headsize_1, headsize_2
        return self._data

    def getlabel(self):
        _, _, _, _, gt_box_1, gt_box_2, _, _, _ = self._fulldata

        labels = []
        b_label = np.ones(gt_box_1.shape[0], dtype=np.float32)
        for batch in range(gt_box_1.shape[0]):
            b_gt_box_1 = gt_box_1[batch].asnumpy()
            grpid_1, subid_1 = b_gt_box_1[4:6]
            b_gt_box_2 = gt_box_2[batch].asnumpy()
            grpid_2, subid_2 = b_gt_box_2[4:6]

            #print(str(int(grpid_1))+'_'+str(int(subid_1)))
            #print(str(int(grpid_2))+'_'+str(int(subid_2)))

            if grpid_1 == grpid_2 and subid_1 == subid_2:
                b_label[batch] *= 1
            else:
                b_label[batch] *= 0

            labels.append(b_label)
            # print(labels)

        labels = mx.nd.array(tensor_vstack(labels, pad=-1))

        self._label = tuple(labels)
        return self._label

    def getindex(self):
        cur_from = self._cur
        cur_to = min(cur_from + self._batch_size, self._size)
        return np.arange(cur_from, cur_to)

    def getpad(self):
        return max(self._cur + self.batch_size - self._size, 0)