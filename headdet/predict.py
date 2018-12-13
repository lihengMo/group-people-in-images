#! /usr/bin/env python

import os
import cv2
from utils import draw_boxes
from frontend import YOLO
import json
import time
from symimdb.groupactivity import GroupActivity
import numpy as np


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
img_list = open('/home/lihengmo/PycharmProjects/test_mxnet/data/GroupActivityPerson/ImageSets/test.txt')
out_path = '/home/lihengmo/head-detection-using-yolo/head/'


def get_head_img(headbox, roi):
    path = '/home/lihengmo/PycharmProjects/test_mxnet/data/GroupActivityPerson/JPEGImages/'
    img_path = path + roi["index"] + '.jpg'
    im = cv2.imread(img_path)

    objs = roi["objs"]
    box_x_difference = 0
    box_y_difference = 0
    box_score = 0
    for box in headbox:
        box.xmin = int(box.xmin * im.shape[1])
        box.ymin = int(box.ymin * im.shape[0])
        box.xmax = int(box.xmax * im.shape[1])
        box.ymax = int(box.ymax * im.shape[0])
        box_x_difference += box.xmax - box.xmin
        box_y_difference += box.ymax - box.ymin
        box_score += box.score

    x_avg = box_x_difference / float(len(headbox))
    y_avg = box_y_difference / float(len(headbox))
    s_avg = box_score / float(len(headbox))

    head_list = np.zeros((len(objs),5),dtype=np.uint16)
    for i in range(len(objs)):
        x,y,xmax,ymax = objs[i]['bbox']
        if x < 0: x = 1
        if xmax > im.shape[1]: xmax = im.shape[1]
        if y < 0: y = 1
        if ymax > im.shape[0]: ymax = im.shape[0]

        tmp_diff = 10000
        tmp_y = 10000
        tmp_score = 0
        tmp_d, tmp_yy, tmp_s = -1,-1,-1
        bestbox = None
        offset_x = x_avg / float(6)
        offset_y = y_avg / float(7)
        gt_box_head_list = []
        for box in headbox:
            if (x-offset_x)< box.xmin and box.xmax< (xmax+offset_x):
                if (y-offset_y)< box.ymin and box.ymax< (ymax+offset_y):
                    if box.score > s_avg*0.5:
                        if box.ymin < (y + (ymax - y) / 13):
                            gt_box_head_list.append(box)

        if len(gt_box_head_list) >= 1:
            gh_score = np.zeros((len(gt_box_head_list)))
            for gh in range(len(gt_box_head_list)):
                    if gt_box_head_list[gh].ymin < tmp_y:
                        tmp_y = gt_box_head_list[gh].ymin
                        tmp_d = gh
                    c_h = (gt_box_head_list[gh].xmin + ((gt_box_head_list[gh].xmax - gt_box_head_list[gh].xmin) / 2))
                    c_b = (x + ((xmax - x) / 2))
                    diff_h = abs(c_b - c_h)
                    if diff_h < tmp_diff:
                        tmp_diff = diff_h
                        tmp_yy = gh
                    if gt_box_head_list[gh].score > tmp_score:
                        tmp_score = gt_box_head_list[gh].score
                        tmp_s = gh
            if tmp_d > -1:
                gh_score[tmp_d] += 1.5
            if tmp_yy > -1:
                gh_score[tmp_yy] += 1
            if tmp_s > -1:
                gh_score[tmp_s] += 1

            if np.max(gh_score) > 0:
                best_index = np.where(gh_score == np.max(gh_score))[0].item()
                bestbox = gt_box_head_list[best_index]

        if bestbox != None:
            head_list[i,:] = [i, bestbox.xmin,bestbox.ymin,bestbox.xmax,bestbox.ymax]
            # gt_box_head_list.append(gt_box_head)
        else:
            center_im = x + ((xmax - x) / 2)
            ratio_im = (ymax - y) / (xmax - x)
            if ratio_im > 2:
                box_l = (ymax - y) / 6.5
                box_xmin = center_im-(box_l/2)
                box_ymin = (y + (ymax - y) / 25)
                head_list[i,:] = [i, box_xmin, box_ymin, box_xmin+box_l, box_ymin+box_l]
            if 1 <= ratio_im and ratio_im <= 2:
                box_l = (ymax - y) / 4.5
                box_xmin = center_im - (box_l / 2)
                box_ymin = (y + (ymax - y) / 25)
                head_list[i, :] = [i, box_xmin, box_ymin, box_xmin + box_l, box_ymin + box_l]
            if ratio_im < 1:
                box_l = (ymax - y) / 3.5
                box_xmin = center_im - (box_l / 2)
                box_ymin = (y + (ymax - y) / 25)
                head_list[i, :] = [i, box_xmin, box_ymin, box_xmin + box_l, box_ymin + box_l]

        bdbox = im[int(y):int(ymax), int(x):int(xmax)]
        bdbox_c = bdbox.copy()
        # hdbox = im[int(box.ymin):int(box.ymax), int(box.xmin):int(box.xmax)]
        i_index,hxmin,hymin,hxmax,hymax = head_list[i,:]
        cv2.rectangle(bdbox_c, (int(hxmin-x),int(hymin-y)), (int(hxmax-x),int(hymax-y)), (0, 255, 0), 2)
        out_path_file = '/home/lihengmo/head-detection-using-yolo/headsin/' + roi['index'] + str(x) + str(
            y) + '_detected.jpg'
        cv2.imwrite(out_path_file, bdbox_c)
        cv2.imwrite('test.jpg',im)

    roi_head_list = {'index': roi['index'],
                     'headbox': head_list}

    return roi_head_list


def get_detect_img(roi, yolo, config):
    path = '/home/lihengmo/PycharmProjects/test_mxnet/data/GroupActivityPerson/JPEGImages/'
    img_path = path + roi["index"] + '.jpg'
    im = cv2.imread(img_path)
    headbox = yolo.predict(im)
    print(len(headbox))
    #image = draw_boxes(im, headbox, config['model']['labels'])
    #out_path_file = '/home/lihengmo/head-detection-using-yolo/headsin/' + roi['index'] + '_detected.jpg'
    #cv2.imwrite(out_path_file, image)
    return headbox


def get_groupactivity(imageset):
    if not imageset:
        imageset = '2018_trainval'

    classes = len(GroupActivity.classes)
    roidb = []
    imdb = GroupActivity(imageset, '/home/lihengmo/PycharmProjects/test_mxnet/data', '/home/lihengmo/PycharmProjects/test_mxnet/data/GroupActivityPerson')
    imdb.filter_roidb()
    roidb.extend(imdb.roidb)

    return roidb


def _main_():
    config_path  = '/home/lihengmo/head-detection-using-yolo/config.json'
    weights_path = '/home/lihengmo/head-detection-using-yolo/model.h5'
    path   = '/home/lihengmo/PycharmProjects/test_mxnet/data/GroupActivityPerson/JPEGImages/'

    roidb = get_groupactivity("2018_trainval")
    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    all_roi_list = []
    for roi in roidb:
        headbox = get_detect_img(roi, yolo, config)
        roi_head_list = get_head_img(headbox, roi)
        all_roi_list.append(roi_head_list)

    print(len(all_roi_list))


if __name__ == '__main__':
    _main_()
