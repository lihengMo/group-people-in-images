import numpy as np
import cv2
import os
import pickle
from symimdb.groupactivity import GroupActivity

img_pixel_means = [0.485, 0.456, 0.406]
img_pixel_stds = [0.229, 0.224, 0.225]


def get_groupactivity(imageset):
    if not imageset:
        imageset = '2018_trainval'

    classes = len(GroupActivity.classes)
    roidb = []
    imdb = GroupActivity(imageset, './data', './data/GroupActivityPerson')
    imdb.filter_roidb()
    roidb.extend(imdb.roidb)

    return roidb


def resize_pair_im(im, resize_h, resize_w):
    im_size_h = im.shape[0]
    im_size_w = im.shape[1]
    im_scale_h = float(resize_h) / float(im_size_h)
    im_scale_w = float(resize_w) / float(im_size_w)
    im = cv2.resize(im, None, None, fx=im_scale_w, fy=im_scale_h, interpolation=cv2.INTER_LINEAR)
    return im, im_scale_h, im_scale_w


def location_map_im(im_shape, gt_box):
    img_h = im_shape[0]
    img_w = im_shape[1]
    im = np.zeros([img_h,img_w,3], dtype=np.uint8)

    x,y,x_max,y_max = gt_box[0:4]
    x = int(x)
    y = int(y)
    x_max = int(x_max)
    y_max = int(y_max)
    if x < 0: x = 1
    if y < 0: y = 1
    if x_max > img_w : x_max = img_w - 1
    if y_max > img_h : y_max = img_h - 1

    im[y:y_max, x:x_max,:] = np.ones([y_max-y,x_max-x,3])*255
    im = cv2.GaussianBlur(im,(9,9),0)
    im = cv2.GaussianBlur(im,(25,25),0)
    return im

def location_map_heads_ori(im_shape, head1,head2):
    im = np.zeros([im_shape[0], im_shape[1], 3], dtype=np.uint8)

    hx, hy, hxmax, hymax = head1[0:4]

    if hx < 0: hx = 1
    if hy < 0: hy = 1
    if hxmax > im_shape[1]: hxmax = im_shape[1] - 1
    if hymax > im_shape[0]: hymax = im_shape[0] - 1

    h = hymax - hy
    w = hxmax - hx

    if im[hy:hymax, hx:hxmax, :].shape[0] != h:
        h = im[hy:hymax, hx:hxmax, :].shape[0]
    if im[hy:hymax, hx:hxmax, :].shape[1] != w:
        w = im[hy:hymax, hx:hxmax, :].shape[1]

    hx1, hy1, hxmax1, hymax1 = head2[0:4]

    if hx1 < 0: hx1 = 1
    if hy1 < 0: hy1 = 1
    if hxmax1 > im_shape[1]: hxmax1 = im_shape[1] - 1
    if hymax1 > im_shape[0]: hymax1 = im_shape[0] - 1

    h1 = hymax1 - hy1
    w1 = hxmax1 - hx1

    if im[hy1:hymax1, hx1:hxmax1, :].shape[0] != h1:
        h1 = im[hy1:hymax1, hx1:hxmax1, :].shape[0]
    if im[hy1:hymax1, hx1:hxmax1, :].shape[1] != w1:
        w1 = im[hy1:hymax1, hx1:hxmax1, :].shape[1]

    if w <= w1:
        cv2.rectangle(im, (hx, hy), (hxmax, hymax), [255, 0, 0], -1)
        b = np.array([[[hx, int(hy + h / 2)], [hx1, hy1], [hxmax1, hy1], [hxmax, int(hy + h / 2)]]], dtype=np.int32)
        cv2.fillPoly(im, b, [0, 255, 0], lineType=cv2.LINE_AA)
        b_b = np.array([[[hx, int(hy + h / 2)], [hx1, hymax1], [hxmax1, hymax1], [hxmax, int(hy + h / 2)]]],
                       dtype=np.int32)
        cv2.fillPoly(im, b_b, [0, 255, 0], lineType=cv2.LINE_AA)
        b_c = np.array([[[hx, int(hy + h / 2)], [hx1, hy1], [hx1, hymax1]]], dtype=np.int32)
        cv2.fillPoly(im, b_c, [0, 255, 0], lineType=cv2.LINE_AA)
        b_d = np.array([[[hxmax, int(hy + h / 2)], [hx1, hy1], [hx1, hymax1]]], dtype=np.int32)
        cv2.fillPoly(im, b_d, [0, 255, 0], lineType=cv2.LINE_AA)
        cv2.rectangle(im, (hx1, hy1), (hxmax1, hymax1), [0, 0, 255], -1)

    if w > w1:
        cv2.rectangle(im, (hx1, hy1), (hxmax1, hymax1), [255, 0, 0], -1)
        b = np.array([[[hx1, int(hy1 + h1 / 2)], [hx, hy], [hxmax, hy], [hxmax1, int(hy1 + h1 / 2)]]], dtype=np.int32)
        cv2.fillPoly(im, b, [0, 255, 0], lineType=cv2.LINE_AA)
        b_b = np.array([[[hx1, int(hy1 + h1 / 2)], [hx, hymax], [hxmax, hymax], [hxmax1, int(hy1 + h1 / 2)]]],
                       dtype=np.int32)
        cv2.fillPoly(im, b_b, [0, 255, 0], lineType=cv2.LINE_AA)
        b_c = np.array([[[hx1, int(hy1 + h1 / 2)], [hx, hy], [hx, hymax]]], dtype=np.int32)
        cv2.fillPoly(im, b_c, [0, 255, 0], lineType=cv2.LINE_AA)
        b_d = np.array([[[hxmax1, int(hy1 + h1 / 2)], [hx, hy], [hx, hymax]]], dtype=np.int32)
        cv2.fillPoly(im, b_d, [0, 255, 0], lineType=cv2.LINE_AA)
        cv2.rectangle(im, (hx, hy), (hxmax, hymax), [0, 0, 255], -1)

    # im = cv2.GaussianBlur(im,(3,3),0)
    return im


def location_map_heads_t(im_shape, head1,head2, ori_im):
    im = np.zeros([im_shape[0], im_shape[1], 3], dtype=np.uint8)

    hx, hy, hxmax, hymax = head1[0:4]

    if hx < 0: hx = 1
    if hy < 0: hy = 1
    if hxmax > im_shape[1]: hxmax = im_shape[1] - 1
    if hymax > im_shape[0]: hymax = im_shape[0] - 1

    h = hymax - hy
    w = hxmax - hx

    if im[hy:hymax, hx:hxmax, :].shape[0] != h:
        h = im[hy:hymax, hx:hxmax, :].shape[0]
    if im[hy:hymax, hx:hxmax, :].shape[1] != w:
        w = im[hy:hymax, hx:hxmax, :].shape[1]

    hx1, hy1, hxmax1, hymax1 = head2[0:4]

    if hx1 < 0: hx1 = 1
    if hy1 < 0: hy1 = 1
    if hxmax1 > im_shape[1]: hxmax1 = im_shape[1] - 1
    if hymax1 > im_shape[0]: hymax1 = im_shape[0] - 1

    h1 = hymax1 - hy1
    w1 = hxmax1 - hx1

    if im[hy1:hymax1, hx1:hxmax1, :].shape[0] != h1:
        h1 = im[hy1:hymax1, hx1:hxmax1, :].shape[0]
    if im[hy1:hymax1, hx1:hxmax1, :].shape[1] != w1:
        w1 = im[hy1:hymax1, hx1:hxmax1, :].shape[1]

    if w <= w1:
        # cv2.rectangle(im, (hx, hy), (hxmax, hymax), [255, 0, 0], -1)
        if hx1 <= hx:
            im[hy:hymax, hx:hxmax] = ori_im[hy:hymax, hx:hxmax]
            b = np.array([[[hx, int(hy + h / 2)], [hx1, hy1], [hxmax1, hy1]]], dtype=np.int32)
            cv2.fillPoly(im, b, [0, 255, 0], lineType=cv2.LINE_AA)
            b_b = np.array([[[hx, int(hy + h / 2)], [hx1, hymax1], [hxmax1, hymax1]]],
                           dtype=np.int32)
            cv2.fillPoly(im, b_b, [0, 255, 0], lineType=cv2.LINE_AA)
            b_c = np.array([[[hx, int(hy + h / 2)], [hx1, hy1], [hx1, hymax1]]], dtype=np.int32)
            cv2.fillPoly(im, b_c, [0, 255, 0], lineType=cv2.LINE_AA)
            b_d = np.array([[[hx, int(hy + h / 2)], [hx1, hy1], [hx1, hymax1]]], dtype=np.int32)
            cv2.fillPoly(im, b_d, [0, 255, 0], lineType=cv2.LINE_AA)
            # cv2.rectangle(im, (hx1, hy1), (hxmax1, hymax1), [0, 0, 255], -1)
            im[hy1:hymax1, hx1:hxmax1] = ori_im[hy1:hymax1, hx1:hxmax1]

        if hx1 > hx:
            im[hy:hymax, hx:hxmax] = ori_im[hy:hymax, hx:hxmax]
            b = np.array([[[hxmax, int(hy + h / 2)], [hx1, hy1], [hxmax1, hy1]]], dtype=np.int32)
            cv2.fillPoly(im, b, [0, 255, 0], lineType=cv2.LINE_AA)
            b_b = np.array([[[hxmax, int(hy + h / 2)], [hx1, hymax1], [hxmax1, hymax1]]],
                           dtype=np.int32)
            cv2.fillPoly(im, b_b, [0, 255, 0], lineType=cv2.LINE_AA)
            b_c = np.array([[[hxmax, int(hy + h / 2)], [hx1, hy1], [hx1, hymax1]]], dtype=np.int32)
            cv2.fillPoly(im, b_c, [0, 255, 0], lineType=cv2.LINE_AA)
            b_d = np.array([[[hxmax, int(hy + h / 2)], [hx1, hy1], [hx1, hymax1]]], dtype=np.int32)
            cv2.fillPoly(im, b_d, [0, 255, 0], lineType=cv2.LINE_AA)
            # cv2.rectangle(im, (hx1, hy1), (hxmax1, hymax1), [0, 0, 255], -1)
            im[hy1:hymax1, hx1:hxmax1] = ori_im[hy1:hymax1, hx1:hxmax1]

    if w > w1:

        if hx <= hx1:
        # cv2.rectangle(im, (hx1, hy1), (hxmax1, hymax1), [255, 0, 0], -1)
            im[hy1:hymax1, hx1:hxmax1] = ori_im[hy1:hymax1, hx1:hxmax1]
            b = np.array([[[hx1, int(hy1 + h1 / 2)], [hx, hy], [hxmax, hy]]], dtype=np.int32)
            cv2.fillPoly(im, b, [0, 255, 0], lineType=cv2.LINE_AA)
            b_b = np.array([[[hx1, int(hy1 + h1 / 2)], [hx, hymax], [hxmax, hymax]]],
                           dtype=np.int32)
            cv2.fillPoly(im, b_b, [0, 255, 0], lineType=cv2.LINE_AA)
            b_c = np.array([[[hx1, int(hy1 + h1 / 2)], [hx, hy], [hx, hymax]]], dtype=np.int32)
            cv2.fillPoly(im, b_c, [0, 255, 0], lineType=cv2.LINE_AA)
            b_d = np.array([[[hx1, int(hy1 + h1 / 2)], [hx, hy], [hx, hymax]]], dtype=np.int32)
            cv2.fillPoly(im, b_d, [0, 255, 0], lineType=cv2.LINE_AA)
            # cv2.rectangle(im, (hx, hy), (hxmax, hymax), [0, 0, 255], -1)
            im[hy:hymax, hx:hxmax] = ori_im[hy:hymax, hx:hxmax]

        if hx > hx1:
        # cv2.rectangle(im, (hx1, hy1), (hxmax1, hymax1), [255, 0, 0], -1)
            im[hy1:hymax1, hx1:hxmax1] = ori_im[hy1:hymax1, hx1:hxmax1]
            b = np.array([[[hxmax1, int(hy1 + h1 / 2)], [hx, hy], [hxmax, hy], [hxmax1, int(hy1 + h1 / 2)]]], dtype=np.int32)
            cv2.fillPoly(im, b, [0, 255, 0], lineType=cv2.LINE_AA)
            b_b = np.array([[[hxmax1, int(hy1 + h1 / 2)], [hx, hymax], [hxmax, hymax], [hxmax1, int(hy1 + h1 / 2)]]],
                           dtype=np.int32)
            cv2.fillPoly(im, b_b, [0, 255, 0], lineType=cv2.LINE_AA)
            b_c = np.array([[[hxmax1, int(hy1 + h1 / 2)], [hx, hy], [hx, hymax]]], dtype=np.int32)
            cv2.fillPoly(im, b_c, [0, 255, 0], lineType=cv2.LINE_AA)
            b_d = np.array([[[hxmax1, int(hy1 + h1 / 2)], [hx, hy], [hx, hymax]]], dtype=np.int32)
            cv2.fillPoly(im, b_d, [0, 255, 0], lineType=cv2.LINE_AA)
            # cv2.rectangle(im, (hx, hy), (hxmax, hymax), [0, 0, 255], -1)
            im[hy:hymax, hx:hxmax] = ori_im[hy:hymax, hx:hxmax]

    return im


def get_test_image(im,roi_rec,roi_head, gt_index, mean,std):
    # im = imdecode(roi_rec['image'])
    if roi_rec["flipped"]:
        im = im[:, ::-1, :]
    im_resize, im_scale_h, im_scale_w = resize_pair_im(im, 250, 375)
    # gt_box_1
    gt_box_1 = np.empty(6, dtype=np.float32)
    gt_box_1[0:4] = roi_rec['boxes'][gt_index[0]]

    grpnum = roi_rec['objs'][gt_index[0]]
    grpids = grpnum['groupnum']
    grpids_list = grpids.split('_')
    grpid, subgrpid = int(grpids_list[0]), int(grpids_list[1])
    gt_box_1[4] = grpid
    gt_box_1[5] = subgrpid

    # gt_box_2
    gt_box_2 = np.empty(6, dtype=np.float32)
    gt_box_2[0:4] = roi_rec['boxes'][gt_index[1]]

    grpnum = roi_rec['objs'][gt_index[1]]
    grpids = grpnum['groupnum']
    grpids_list = grpids.split('_')
    grpid, subgrpid = int(grpids_list[0]), int(grpids_list[1])
    gt_box_2[4] = grpid
    gt_box_2[5] = subgrpid

    # pair box images
    x, y, x_max, y_max = gt_box_1[0:4]
    pair_im_1 = im[int(y):int(y_max), int(x):int(x_max)]
    pair_im_1, _, _ = resize_pair_im(pair_im_1, 250, 100)
    # pair_im_1_tensor = transform_resnet(pair_im_1, mean, std)

    x, y, x_max, y_max = gt_box_2[0:4]
    pair_im_2 = im[int(y):int(y_max), int(x):int(x_max)]
    pair_im_2, _, _ = resize_pair_im(pair_im_2, 250, 100)
    # pair_im_2_tensor = transform_resnet(pair_im_2, mean, std)

    # scale gt_boxes
    gt_box_1[0] *= im_scale_w
    gt_box_1[2] *= im_scale_w
    gt_box_1[1] *= im_scale_h
    gt_box_1[3] *= im_scale_h

    gt_box_2[0] *= im_scale_w
    gt_box_2[2] *= im_scale_w
    gt_box_2[1] *= im_scale_h
    gt_box_2[3] *= im_scale_h

    # location map of pair image

    location_map_1 = location_map_im(im_resize.shape, gt_box_1)
    # location_map_1_tensor = transform_map(location_map_1)
    location_map_2 = location_map_im(im_resize.shape, gt_box_2)
    # location_map_2_tensor = transform_map(location_map_2)

    head_1 = roi_head['headbox'][gt_index[0]]
    head_2 = roi_head['headbox'][gt_index[1]]

    x1, y1, xmax1, ymax1 = head_1[1:5]
    x2, y2, xmax2, ymax2 = head_2[1:5]

    x1 *= im_scale_w
    xmax1 *= im_scale_w
    y1 *= im_scale_h
    ymax1 *= im_scale_h

    x2 *= im_scale_w
    xmax2 *= im_scale_w
    y2 *= im_scale_h
    ymax2 *= im_scale_h

    h1 = np.array((int(x1), int(y1), int(xmax1), int(ymax1)))
    h2 = np.array((int(x2), int(y2), int(xmax2), int(ymax2)))

    r_map = location_map_heads_t(im_resize.shape,h1,h2,im_resize)
    folder = "./head_mask/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = folder + roi_rec["index"] + str(gt_index[0]) + '_' + str(gt_index[1]) + ".jpg"
    cv2.imwrite(path,r_map)
    r_map_ori = location_map_heads_ori(im_resize.shape, h1, h2)
    path = folder + roi_rec["index"] + str(gt_index[0]) + '_' + str(gt_index[1]) + "_ori.jpg"
    # cv2.imwrite(path,r_map_ori)
    # r_map_tensor = transform_map(r_map)

    headsize_1 = np.array([(ymax1-y1), (xmax1-x1)])
    # print(headsize_1)
    headsize_2 = np.array([(ymax2-y2), (xmax2-x2)])
    # print(headsize_2)

# load dataset
imageset = '2018_trainval'
roidb = get_groupactivity(imageset)

# load head detection
cache_path = './data/cache/roi_head.pkl'
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as fid:
        cached = pickle.load(fid)
        roihead = cached

for index in range(len(roidb)):

    if index != None:
        roidb_batch = roidb[index]
        roihead_batch = roihead[index]

        if roihead_batch['index'] != None:

            im = cv2.imread(roidb_batch['image'])

            if roidb_batch['gt_classes'].size > 0:
                gt_inds = np.where(roidb_batch['gt_classes'] != 0)[0]

            from itertools import combinations
            gt_indexs = list(combinations(gt_inds, 2))

            for cur in range(len(gt_indexs)):
                gt_index = gt_indexs[cur]
                get_test_image(im,roidb_batch,roihead_batch,gt_index,img_pixel_means,img_pixel_stds)




