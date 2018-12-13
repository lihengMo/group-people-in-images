import mxnet as mx

eps=1e-5
use_global_stats = False
workspace=1024

def get_feature(data):
    # first conv layer
    conv1 = mx.sym.Convolution(data=data, kernel=(5, 5), num_filter=20)
    tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
    pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2, 2), stride=(2, 2))
    # second conv layer
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), num_filter=50)
    tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
    pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2, 2), stride=(2, 2))

    return pool2

def get_map_feature(data):
    conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(3, 3),
                                       kernel=(7, 7),
                                       stride=(2, 2), no_bias=False)
    ReLU1 = mx.symbol.LeakyReLU(name='ReLU1', data=conv1, act_type='leaky', slope=0.01)
    conv2 = mx.symbol.Convolution(name='conv2', data=ReLU1, num_filter=128, pad=(2, 2), kernel=(5, 5),
                                  stride=(2, 2),
                                  no_bias=False)
    ReLU2 = mx.symbol.LeakyReLU(name='ReLU2', data=conv2, act_type='leaky', slope=0.01)
    conv3 = mx.symbol.Convolution(name='conv3', data=ReLU2, num_filter=256, pad=(2, 2), kernel=(5, 5),
                                  stride=(2, 2),
                                  no_bias=False)
    ReLU3 = mx.symbol.LeakyReLU(name='ReLU3', data=conv3, act_type='leaky', slope=0.01)
    conv3_1 = mx.symbol.Convolution(name='conv3_1', data=ReLU3, num_filter=256, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    ReLU4 = mx.symbol.LeakyReLU(name='ReLU4', data=conv3_1, act_type='leaky', slope=0.01)
    conv4 = mx.symbol.Convolution(name='conv4', data=ReLU4, num_filter=512, pad=(1, 1), kernel=(3, 3),
                                  stride=(2, 2),
                                  no_bias=False)
    ReLU5 = mx.symbol.LeakyReLU(name='ReLU5', data=conv4, act_type='leaky', slope=0.01)
    pool1 = mx.sym.Pooling(data=ReLU5, pool_type="max", kernel=(2, 2), stride=(2, 2))
    return pool1


def residual_unit(data, num_filter, stride, dim_match, name):
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn3')
    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                      workspace=workspace, name=name + '_sc')
    sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')
    return sum


def get_resnet_feature(data, units, filter_list):
    # res1
    # data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='bn_data')
    conv0 = mx.sym.Convolution(data=data, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1')
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True, name='stage1_unit%s' % i)

    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1')
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True, name='stage2_unit%s' % i)

    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1')
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True, name='stage3_unit%s' % i)

    unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i)
    bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    return pool1

def get_resnet_top_feature(data, units, filter_list):
    # res1
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='bn_data_t')
    conv0 = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv_t", workspace=workspace)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn0_t')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0_t')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0_t')

    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='t_stage_unit1')

    bn2 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn2_t')
    relu2 = mx.sym.Activation(data=bn2, act_type='relu', name='relu1_t')
    pool2 = mx.symbol.Pooling(data=relu2, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool2_t')
    return pool2


def get_resnet_train_symbol(units,filter_list):
    pair_im_1 = mx.sym.var('pair_im_1')
    pair_im_2 = mx.sym.var('pair_im_2')
    loca_map_1 = mx.sym.var('loca_map_1')
    loca_map_2 = mx.sym.var('loca_map_2')
    r_map = mx.sym.var('r_map')
    headsize1 = mx.sym.var('headsize_1')
    headsize2 = mx.sym.var('headsize_2')
    pair_label = mx.sym.var('label')

    concat_data = mx.symbol.Concat(*[pair_im_1,pair_im_2], dim=0)
    concat_map_data = mx.symbol.Concat(loca_map_1,loca_map_2,r_map, dim=0)
    pool2 = get_resnet_feature(concat_data,units,filter_list)
    map_feature = get_map_feature(concat_map_data)
    pair_feats = mx.sym.SliceChannel(data=pool2,axis=0, num_outputs=2)
    map_feats = mx.sym.SliceChannel(data=map_feature,axis=0, num_outputs=3)

    # head_feats_1 = get_resnet_top_feature(r_map,units,filter_list)

    per_feat_1 = mx.sym.flatten(data=pair_feats[0])
    per_feat_2 = mx.sym.flatten(data=pair_feats[1])
    map_feat_1 = mx.sym.flatten(data=map_feats[0])
    map_feat_2 = mx.sym.flatten(data=map_feats[1])
    head_feat = mx.sym.flatten(data=map_feats[2])
    # f_head_feats_1 = mx.sym.flatten(data=head_feats_1)
    headsize1_feats = mx.sym.flatten(data=headsize1)
    headsize2_feats = mx.sym.flatten(data=headsize2)

    concat_h_feat_1 = mx.sym.concat(head_feat, headsize1_feats,headsize2_feats,dim=1)

    concat_feat_1 = mx.sym.concat(per_feat_1, map_feat_1,dim=1)
    concat_feat_2 = mx.sym.concat(per_feat_2, map_feat_2,dim=1)

    # sum_feat = mx.symbol.ElementWiseSum(concat_feat_1, concat_feat_2)
    concat_feat = mx.sym.concat(concat_feat_1,concat_feat_2, concat_h_feat_1, dim=1)

    # first fullc layer
    fc1 = mx.symbol.FullyConnected(data=concat_feat, num_hidden=500)
    reluf = mx.sym.Activation(data=fc1, act_type="relu")
    # second fullc
    fc2 = mx.sym.FullyConnected(data=reluf, num_hidden=2)
    # softmax loss
    net = mx.sym.SoftmaxOutput(data=fc2,label=pair_label, normalization='batch', use_ignore=True,
                                 ignore_label=-1, name='net')
    return net