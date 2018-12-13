import json
from headdet.gen_head import get_head_img
import pickle
import time

def head_detection(roidb, save_name):
    print('can not find head detection cache, now is running the head detection.')
    from headdet.frontend import YOLO
    config_path = './headdet/config.json'
    weights_path = './headdet/model.h5'

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Make the model
    ###############################

    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes
    ###############################

    all_roi_head_list = []
    start = time.time()
    print('detecting the head...')
    for roi in roidb:
        roi_head_list = get_head_img(yolo, roi)
        all_roi_head_list.append(roi_head_list)
    end = time.time()
    t = end - start
    print('has detected: ' + str(len(all_roi_head_list)) + ' images.')
    print('the total detected time: ' + str(t))
    cached = all_roi_head_list
    cache_path = './data/cache/' + save_name + '.pkl'
    with open(cache_path, 'wb') as fid:
        pickle.dump(cached, fid, pickle.HIGHEST_PROTOCOL)

    return all_roi_head_list