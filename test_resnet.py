import mxnet as mx
import os
import numpy as np
import cv2
from symimdb.groupactivity import GroupActivity
from loaddata.Loader import Loader, load_test, generate_batch
import time
from matplotlib import pyplot as plt
from headdet.gen_head import get_head_img
from symnet.symbol_net_test import get_resnet_test_symbol
from symnet.model import load_param, check_shape
import json
import pickle
import sys
from cluster.cluster_tool import group_clustering
from headdet.head_detection import head_detection

res_file = '{}_{}.log'.format('./output/result_log', time.strftime('%Y-%m-%d-%H-%M'))
res = open(res_file,'a')

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

def get_groupactivity(imageset):
    if not imageset:
        imageset = '2018_trainval'

    classes = len(GroupActivity.classes)

    roidb = []
    imdb = GroupActivity(imageset, './data', './data/GroupActivityPerson')
    imdb.filter_roidb()
    roidb.extend(imdb.roidb)

    return roidb

# test model path & parameter
params_sym = './model/test/pairnet-resnet-0047.params'
img_pixel_means = [0.485, 0.456, 0.406]
img_pixel_stds = [0.229, 0.224, 0.225]
resnet_units = (3, 4, 6, 3)
resnet_filter_list = (256, 512, 1024, 2048)

# for doing the cluster
do_cluster = True
given_cluster_num = True
cut_matrix_threshold = 0.45
vis = True
vis_cluster_outpath='./cluster_result/'
if do_cluster:
    print('will do group clustering while testing.')
    if not os.path.exists(vis_cluster_outpath):
        os.makedirs(vis_cluster_outpath)

# load labelling roi
imageset = '2018_test'
roidb = get_groupactivity(imageset)

# load head detection
cache_path = './data/cache/roi_head_test.pkl'
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as fid:
        cached = pickle.load(fid)
        roihead_test = cached
else:
    detect_save_name = 'roi_head_test'
    roihead_test = head_detection(roidb, detect_save_name)
    try:
        sys.exit()
    finally:
        print('the head detection has finished and saved in cache. please run the test again.')

# setting network parameter
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

data_names = ['pair_im_1', 'pair_im_2', 'loca_map_1', 'loca_map_2','r_map','headsize_1','headsize_2']
label_names = None


data_shapes = [('pair_im_1', (1,3,250,100)), ('pair_im_2', (1,3,250,100)),
               ('loca_map_1', (1,3,250,375)), ('loca_map_2', (1,3,250,375)),
               ('r_map',  (1,3,250,375)), ('headsize_1',  (1,2)), ('headsize_2', (1,2))]
label_shapes = None


# loading symbol model
sym = get_resnet_test_symbol(resnet_units, resnet_filter_list)
arg_params, aux_params = load_param(params_sym, ctx=ctx)
check_shape(sym,data_shapes,arg_params,aux_params)

mod = mx.mod.Module(symbol=sym, context=ctx, data_names=data_names, label_names=label_names)
mod.bind(data_shapes, label_shapes, for_training=False)
mod.init_params(arg_params=arg_params, aux_params=aux_params)

# test
tp_all, fp_all = 0, 0
acc_all = 0.0
for index_roi in range(len(roidb)):
    start = time.time()
    roidb_batch = roidb[index_roi]
    roihead = roihead_test[index_roi]

    print(roidb_batch)
    print('roi[{0}]:{1}'.format(index_roi,roidb_batch),file=res)

    tp, fp = 0, 0
    acc = 0.0
    if roidb_batch['gt_classes'].size > 0:
        gt_inds = np.where(roidb_batch['gt_classes'] != 0)[0]

        im = cv2.imread(roidb_batch['image'])

        if do_cluster:
            objs = roidb_batch['objs']
            n_cluster = 1
            for obj in range(len(objs)):
                length = len(objs)-1
                if obj < length:
                    g_1 = objs[obj]['groupnum']
                    g_2 = objs[obj+1]['groupnum']
                    if g_1 != g_2:
                        n_cluster += 1

            affinity = np.zeros(shape=(len(gt_inds),len(gt_inds)))
            pairwise_list = []

        from itertools import combinations
        gt_indexs = list(combinations(gt_inds, 2))
        for cur in range(len(gt_indexs)):
            gt_index = gt_indexs[cur]

            pair_im_1_tensor, pair_im_2_tensor, location_map_1_tensor, location_map_2_tensor, \
            r_map_tensor,headsize_1,headsize_2,label = load_test(im, roidb_batch,roihead,gt_index,img_pixel_means,img_pixel_stds)
            data_batch = generate_batch(pair_im_1_tensor, pair_im_2_tensor, location_map_1_tensor, location_map_2_tensor,r_map_tensor,headsize_1,headsize_2)

            mod.forward(data_batch)
            prob = mod.get_outputs()
            score = prob[0][0].asnumpy()
            print('gt_index[{}, {}]'.format(gt_index[0],gt_index[1]),file=res)
            print('gt_index[%s, %s]' % (gt_index[0], gt_index[1]))
            print('pred:{0}, gt:{1}'.format(score, label), file=res)
            print('pred: %s, gt:%s' % (score, label))
            score_neg = score[0]
            score_pos = score[1]

            if score_neg > score_pos:
                pred = 0
            else:
                pred = 1

            if do_cluster:
                affinity[gt_index[0]][gt_index[1]] = score_pos
                affinity[gt_index[1]][gt_index[0]] = score_pos
                pairwise = (gt_index[0], gt_index[1], score_neg, score_pos, pred)
                pairwise_list.append(pairwise)

            if pred == label:
                tp += 1
            if pred != label:
                fp += 1

            acc = tp / float(tp + fp)
        tp_all += tp
        fp_all += fp

    print('tp_img:%s , fp_img:%s, acc_img:%f' % (tp,fp,acc))
    print('tp_img:{0}'.format(tp), file=res)
    print('fp_img:{0}'.format(fp), file=res)
    print('acc_img:{0}'.format(acc), file=res)
    end = time.time()
    print('pairwise detecting time: %s' % (end-start))

    if do_cluster:
        gt, pred_labels, score = group_clustering(affinity, roidb, index_roi, n_cluster, pairwise_list,
                                                  cut_thre=cut_matrix_threshold, give_cluster=given_cluster_num,
                                                  vis=vis, path=vis_cluster_outpath)
        print('ground-truth group:{0}'.format(gt), file=res)
        print('grouping prediction:{0}'.format(pred_labels), file=res)
        print('v-measure score:{0}'.format(score), file=res)


acc_all = tp_all / float(tp_all + fp_all)

print('tp:%s , fp:%s, acc:%f' % (tp_all, fp_all, acc_all))
print('tp:{0}'.format(tp_all), file=res)
print('fp:{0}'.format(fp_all), file=res)
print('acc:{0}'.format(acc_all), file=res)
