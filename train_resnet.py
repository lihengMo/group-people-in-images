import mxnet as mx
import os
import numpy as np
from symimdb.groupactivity import GroupActivity
from loaddata.Loader import Loader
from symnet.symbol_net import get_resnet_train_symbol
from symnet.model import load_param, load_param_resnet,\
    check_shape, get_fixed_params,initialize_lenet,initialize_resnet,infer_data_shape,initialize_resnet_full, initialize_resnet_leaky,load_param_resnet_full
from headdet.gen_head import get_head_img
import json
import cv2
import time
import pickle
import sys
from headdet.head_detection import head_detection

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

# params
is_init = True
is_resume = False
img_min_size = 400
img_max_size = 800
img_pixel_means = [0.485, 0.456, 0.406]
img_pixel_stds = [0.229, 0.224, 0.225]
batch_size = 8
resnet_units = (3, 4, 6, 3)
resnet_filter_list = (256, 512, 1024, 2048)
arg_params ={}
aux_params ={}

net_fixed_params = ['conv0','stage1','stage2','stage3','bn0_gamma','bn0_beta']
pretrained = './model/pretrained_model/resnet-50-0000.params'
save_prefix = './model/pairnet-resnet'
resume = './model/new_feats_leaky_normalized_true/pairnet-resnet-0016.params'


def get_groupactivity(imageset):
    if not imageset:
        imageset = '2018_trainval'

    classes = len(GroupActivity.classes)
    roidb = []
    imdb = GroupActivity(imageset, './data', './data/GroupActivityPerson')
    imdb.filter_roidb()
    roidb.extend(imdb.roidb)

    return roidb

# load dataset
imageset = '2018_trainval'
roidb = get_groupactivity(imageset)

# load head detection
cache_path = './data/cache/roi_head.pkl'
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as fid:
        cached = pickle.load(fid)
        roihead = cached
else:
    detect_save_name = 'roi_head'
    roihead = head_detection(roidb, detect_save_name)
    try:
       sys.exit()
    finally:
        print('the head detection has finished and saved in cache. please run the train again.')

# system setting
mx.random.seed(42)
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

# load symbol network
net = get_resnet_train_symbol(units=resnet_units,filter_list=resnet_filter_list)

# load training data
train_data = Loader(roidb, roihead, batch_size, img_pixel_means, img_pixel_stds)
data_shape_dict, out_shape_dict = infer_data_shape(net, train_data.provide_data+train_data.provide_label)

# load params
if is_init:
    arg_params, aux_params = load_param_resnet_full(pretrained, ctx)
    arg_params, aux_params = initialize_resnet_leaky(net,train_data.provide_data,arg_params, aux_params)
    print('initialize and train the new network')
if is_resume:
    arg_params, aux_params = load_param(resume, ctx)
    print('resume from trained-network')

# symbol data
data_names = ['pair_im_1', 'pair_im_2','loca_map_1', 'loca_map_2','r_map','headsize_1','headsize_2']
label_names = ['label']

check_shape(net,train_data.provide_data+train_data.provide_label,arg_params,aux_params)

fixed_param_names = get_fixed_params(net,net_fixed_params)

# callback
batch_end_callback = mx.callback.Speedometer(batch_size, frequent=100, auto_reset=False)
epoch_end_callback = mx.callback.do_checkpoint(save_prefix)

# metric
eval_metrics_1 = mx.metric.Accuracy()
eval_metrics_2 = mx.metric.CrossEntropy()
eval_metrics = mx.metric.CompositeEvalMetric()
for child_metric in [eval_metrics_1, eval_metrics_2]:
    eval_metrics.add(child_metric)


# momentum
base_lr = 0.0008
lr_factor = 0.1
lr_decay_epoch = '10,20,30,40,50,60'
lr_epoch = [int(epoch) for epoch in lr_decay_epoch.split(',')]
lr_epoch_diff = [epoch - 0 for epoch in lr_epoch if epoch > 0]
lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)

lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)
# lr_scheduler = mx.lr_scheduler.PolyScheduler(max_update=25,base_lr=base_lr,pwr=2)

# optimizer
optimizer_params = {'momentum': 0.9,
                    'wd': 0.0005,
                    'learning_rate': lr,
                    'lr_scheduler': lr_scheduler,
                    'rescale_grad': 1.0,
                    'clip_gradient': 5.0
                    }


# train
net_model = mx.mod.Module(symbol=net, data_names=data_names, label_names=label_names, fixed_param_names=fixed_param_names,context=ctx)

net_model.fit(train_data,
                optimizer='sgd',
                optimizer_params=optimizer_params,
                eval_metric=eval_metrics,
                epoch_end_callback=epoch_end_callback,
                batch_end_callback=batch_end_callback,
                arg_params=arg_params, aux_params=aux_params,
                num_epoch=70)



