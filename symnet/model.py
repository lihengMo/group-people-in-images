import mxnet as mx


def load_param(params, ctx=None):
    """same as mx.model.load_checkpoint, but do not load symnet and will convert context"""
    if ctx is None:
        ctx = mx.cpu()
    save_dict = mx.nd.load(params)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v.as_in_context(ctx)
        if tp == 'aux':
            aux_params[name] = v.as_in_context(ctx)
    return arg_params, aux_params


def load_param_resnet_full(params, ctx=None):
    """same as mx.model.load_checkpoint, but do not load symnet and will convert context"""
    if ctx is None:
        ctx = mx.cpu()
    save_dict = mx.nd.load(params)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if name != 'fc1_weight':
            if name != 'fc1_bias':
                if tp == 'arg':
                    arg_params[name] = v.as_in_context(ctx)
                if tp == 'aux':
                    aux_params[name] = v.as_in_context(ctx)
    return arg_params, aux_params


def load_param_resnet(params, ctx=None):
    """same as mx.model.load_checkpoint, but do not load symnet and will convert context"""
    need_para =["stage4","bn1_gamma","bn1_beta","bn1_moving_mean","bn1_moving_var"]
    if ctx is None:
        ctx = mx.cpu()
    save_dict = mx.nd.load(params)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        count = 0
        count2 = 0
        if need_para[0] in name :
            count += 1
            count2 += 1

        if need_para[1] == name:
            count += 1

        if need_para[2] == name:
            count += 1

        if need_para[3] == name:
            count2 += 1

        if need_para[4] == name:
            count2 += 1


        if count > 0:
            if tp == 'arg':
                arg_params[name] = v.as_in_context(ctx)

        if count2 > 0:
            if tp == 'aux':
                aux_params[name] = v.as_in_context(ctx)


    return arg_params, aux_params


def infer_param_shape(symbol, data_shapes):
    arg_shape, _, aux_shape = symbol.infer_shape(**dict(data_shapes))
    arg_shape_dict = dict(zip(symbol.list_arguments(), arg_shape))
    aux_shape_dict = dict(zip(symbol.list_auxiliary_states(), aux_shape))
    return arg_shape_dict, aux_shape_dict


def infer_data_shape(symbol, data_shapes):
    _, out_shape, _ = symbol.infer_shape(**dict(data_shapes))
    data_shape_dict = dict(data_shapes)
    out_shape_dict = dict(zip(symbol.list_outputs(), out_shape))
    return data_shape_dict, out_shape_dict


def check_shape(symbol, data_shapes, arg_params, aux_params):
    arg_shape_dict, aux_shape_dict = infer_param_shape(symbol, data_shapes)
    data_shape_dict, out_shape_dict = infer_data_shape(symbol, data_shapes)
    for k in symbol.list_arguments():
        if k in data_shape_dict or 'label' in k:
            continue
        assert k in arg_params, '%s not initialized' % k
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for %s inferred %s provided %s' % (k, arg_shape_dict[k], arg_params[k].shape)
    for k in symbol.list_auxiliary_states():
        assert k in aux_params, '%s not initialized' % k
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for %s inferred %s provided %s' % (k, aux_shape_dict[k], aux_params[k].shape)

def initialize_params(symbol,data_shapes, arg_params, aux_params):
    arg_shape_dict, aux_shape_dict = infer_param_shape(symbol, data_shapes)
    arg_params['convolution0_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['convolution0_weight'])
    arg_params['convolution0_bias'] = mx.nd.zeros(shape=arg_shape_dict['convolution0_bias'])
    arg_params['convolution1_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['convolution1_weight'])
    arg_params['convolution1_bias'] = mx.nd.zeros(shape=arg_shape_dict['convolution1_bias'])
    arg_params['fc1_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fc1_weight'])
    arg_params['fc1_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc1_bias'])
    arg_params['fc2_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fc2_weight'])
    arg_params['fc2_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc2_bias'])
    return arg_params, aux_params

def initialize_lenet(symbol, data_shapes, arg_params, aux_params):
    arg_shape_dict, aux_shape_dict = infer_param_shape(symbol, data_shapes)
    arg_params['convolution4_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['convolution4_weight'])
    arg_params['convolution4_bias'] = mx.nd.zeros(shape=arg_shape_dict['convolution4_bias'])
    arg_params['convolution5_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['convolution5_weight'])
    arg_params['convolution5_bias'] = mx.nd.zeros(shape=arg_shape_dict['convolution5_bias'])

    arg_params['fullyconnected0_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fullyconnected0_weight'])
    arg_params['fullyconnected0_bias'] = mx.nd.zeros(shape=arg_shape_dict['fullyconnected0_bias'])
    arg_params['fullyconnected1_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fullyconnected1_weight'])
    arg_params['fullyconnected1_bias'] = mx.nd.zeros(shape=arg_shape_dict['fullyconnected1_bias'])
    return arg_params, aux_params

def initialize_resnet(symbol, data_shapes, arg_params, aux_params):
    arg_shape_dict, aux_shape_dict = infer_param_shape(symbol, data_shapes)

    arg_params['fullyconnected0_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fullyconnected0_weight'])
    arg_params['fullyconnected0_bias'] = mx.nd.zeros(shape=arg_shape_dict['fullyconnected0_bias'])
    arg_params['fullyconnected1_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fullyconnected1_weight'])
    arg_params['fullyconnected1_bias'] = mx.nd.zeros(shape=arg_shape_dict['fullyconnected1_bias'])

    arg_params['conv_t_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['conv_t_weight'])
    arg_params['bn0_t_gamma'] = mx.nd.ones(shape=arg_shape_dict['bn0_t_gamma'])
    arg_params['bn0_t_beta'] = mx.nd.zeros(shape=arg_shape_dict['bn0_t_beta'])
    arg_params['t_stage_unit1_bn1_gamma'] = mx.nd.ones(shape=arg_shape_dict['t_stage_unit1_bn1_gamma'])
    arg_params['t_stage_unit1_bn1_beta'] = mx.nd.zeros(shape=arg_shape_dict['t_stage_unit1_bn1_beta'])
    arg_params['t_stage_unit1_conv1_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['t_stage_unit1_conv1_weight'])
    arg_params['t_stage_unit1_bn2_gamma'] = mx.nd.ones(shape=arg_shape_dict['t_stage_unit1_bn2_gamma'])
    arg_params['t_stage_unit1_bn2_beta'] = mx.nd.zeros(shape=arg_shape_dict['t_stage_unit1_bn2_beta'])
    arg_params['t_stage_unit1_conv2_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['t_stage_unit1_conv2_weight'])
    arg_params['t_stage_unit1_bn3_gamma'] = mx.nd.ones(shape=arg_shape_dict['t_stage_unit1_bn3_gamma'])
    arg_params['t_stage_unit1_bn3_beta'] = mx.nd.zeros(shape=arg_shape_dict['t_stage_unit1_bn3_beta'])
    arg_params['t_stage_unit1_conv3_weight'] = mx.random.normal(0, 0.01,shape=arg_shape_dict['t_stage_unit1_conv3_weight'])
    arg_params['t_stage_unit1_sc_weight'] = mx.random.normal(0, 0.01,shape=arg_shape_dict['t_stage_unit1_sc_weight'])
    arg_params['bn2_t_gamma'] = mx.nd.ones(shape=arg_shape_dict['bn2_t_gamma'])
    arg_params['bn2_t_beta'] = mx.nd.zeros(shape=arg_shape_dict['bn2_t_beta'])
    arg_params['bn_data_t_gamma'] = mx.nd.ones(shape=arg_shape_dict['bn_data_t_gamma'])
    arg_params['bn_data_t_beta'] = mx.nd.zeros(shape=arg_shape_dict['bn_data_t_beta'])

    aux_params['bn_data_t_moving_mean'] = mx.nd.zeros(shape=aux_shape_dict['bn_data_t_moving_mean'])
    aux_params['bn_data_t_moving_var'] = mx.nd.ones(shape=aux_shape_dict['bn_data_t_moving_var'])
    aux_params['bn2_t_moving_mean'] = mx.nd.zeros(shape=aux_shape_dict['bn2_t_moving_mean'])
    aux_params['bn2_t_moving_var'] = mx.nd.ones(shape=aux_shape_dict['bn2_t_moving_var'])
    aux_params['bn0_t_moving_mean'] = mx.nd.zeros(shape=aux_shape_dict['bn0_t_moving_mean'])
    aux_params['bn0_t_moving_var'] = mx.nd.ones(shape=aux_shape_dict['bn0_t_moving_var'])
    aux_params['t_stage_unit1_bn1_moving_mean'] = mx.nd.zeros(shape=aux_shape_dict['t_stage_unit1_bn1_moving_mean'])
    aux_params['t_stage_unit1_bn1_moving_var'] = mx.nd.ones(shape=aux_shape_dict['t_stage_unit1_bn1_moving_var'])
    aux_params['t_stage_unit1_bn2_moving_mean'] = mx.nd.zeros(shape=aux_shape_dict['t_stage_unit1_bn2_moving_mean'])
    aux_params['t_stage_unit1_bn2_moving_var'] = mx.nd.ones(shape=aux_shape_dict['t_stage_unit1_bn2_moving_var'])
    aux_params['t_stage_unit1_bn3_moving_mean'] = mx.nd.zeros(shape=aux_shape_dict['t_stage_unit1_bn3_moving_mean'])
    aux_params['t_stage_unit1_bn3_moving_var'] = mx.nd.ones(shape=aux_shape_dict['t_stage_unit1_bn3_moving_var'])

    return arg_params, aux_params


def initialize_resnet_full(symbol, data_shapes, arg_params, aux_params):
    arg_shape_dict, aux_shape_dict = infer_param_shape(symbol, data_shapes)
    arg_params['convolution0_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['convolution0_weight'])
    arg_params['convolution0_bias'] = mx.nd.zeros(shape=arg_shape_dict['convolution0_bias'])
    arg_params['convolution1_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['convolution1_weight'])
    arg_params['convolution1_bias'] = mx.nd.zeros(shape=arg_shape_dict['convolution1_bias'])
    arg_params['convolution2_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['convolution2_weight'])
    arg_params['convolution2_bias'] = mx.nd.zeros(shape=arg_shape_dict['convolution2_bias'])
    arg_params['convolution3_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['convolution3_weight'])
    arg_params['convolution3_bias'] = mx.nd.zeros(shape=arg_shape_dict['convolution3_bias'])

    arg_params['fullyconnected0_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fullyconnected0_weight'])
    arg_params['fullyconnected0_bias'] = mx.nd.zeros(shape=arg_shape_dict['fullyconnected0_bias'])
    arg_params['fullyconnected1_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fullyconnected1_weight'])
    arg_params['fullyconnected1_bias'] = mx.nd.zeros(shape=arg_shape_dict['fullyconnected1_bias'])

    return arg_params, aux_params

def initialize_resnet_leaky(symbol, data_shapes, arg_params, aux_params):
    arg_shape_dict, aux_shape_dict = infer_param_shape(symbol, data_shapes)
    arg_params['conv1_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['conv1_weight'])
    arg_params['conv1_bias'] = mx.nd.zeros(shape=arg_shape_dict['conv1_bias'])
    arg_params['conv2_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['conv2_weight'])
    arg_params['conv2_bias'] = mx.nd.zeros(shape=arg_shape_dict['conv2_bias'])
    arg_params['conv3_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['conv3_weight'])
    arg_params['conv3_bias'] = mx.nd.zeros(shape=arg_shape_dict['conv3_bias'])
    arg_params['conv3_1_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['conv3_1_weight'])
    arg_params['conv3_1_bias'] = mx.nd.zeros(shape=arg_shape_dict['conv3_1_bias'])
    arg_params['conv4_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['conv4_weight'])
    arg_params['conv4_bias'] = mx.nd.zeros(shape=arg_shape_dict['conv4_bias'])

    arg_params['fullyconnected0_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fullyconnected0_weight'])
    arg_params['fullyconnected0_bias'] = mx.nd.zeros(shape=arg_shape_dict['fullyconnected0_bias'])
    arg_params['fullyconnected1_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fullyconnected1_weight'])
    arg_params['fullyconnected1_bias'] = mx.nd.zeros(shape=arg_shape_dict['fullyconnected1_bias'])

    return arg_params, aux_params

def get_fixed_params(symbol, fixed_param_prefix=''):
    fixed_param_names = []
    fixed_param_names_fixed = []
    if fixed_param_prefix:
        for name in symbol.list_arguments():
            for prefix in fixed_param_prefix:
                if prefix in name:
                    fixed_param_names.append(name)

        for i in range(len(fixed_param_names)-1):
            if fixed_param_names[i] != fixed_param_names[i+1]:
                fixed_param_names_fixed.append(fixed_param_names[i])

        fixed_param_names_fixed.append(fixed_param_names[-1])

    return fixed_param_names_fixed
