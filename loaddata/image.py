import numpy as np
import cv2
import random
import mxnet as mx

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
    pair_im_1_tensor = transform_resnet(pair_im_1, mean, std)

    x, y, x_max, y_max = gt_box_2[0:4]
    pair_im_2 = im[int(y):int(y_max), int(x):int(x_max)]
    pair_im_2, _, _ = resize_pair_im(pair_im_2, 250, 100)
    pair_im_2_tensor = transform_resnet(pair_im_2, mean, std)

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
    location_map_1_tensor = transform_map(location_map_1)
    location_map_2 = location_map_im(im_resize.shape, gt_box_2)
    location_map_2_tensor = transform_map(location_map_2)

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
    path = "./testmask/" + roi_rec["index"] + ".jpg"
    # cv2.imwrite(path,r_map)
    r_map_tensor = transform_map(r_map)

    headsize_1 = np.array([(ymax1-y1), (xmax1-x1)])
    # print(headsize_1)
    headsize_2 = np.array([(ymax2-y2), (xmax2-x2)])
    # print(headsize_2)

    return pair_im_1_tensor, pair_im_2_tensor, location_map_1_tensor, location_map_2_tensor, \
           gt_box_1, gt_box_2, r_map_tensor, headsize_1,headsize_2


def get_image(roi_rec, roi_head, mean, std):
    """
    read, resize, transform image, return im_tensor, im_info, gt_boxes
    roi_rec should have keys: ["image", "boxes", "gt_classes", "flipped"]
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    # print(roi_rec['index'])
    im = imdecode(roi_rec['image'])
    if roi_rec["flipped"]:
        im = im[:, ::-1, :]
    im_resize, im_scale_h, im_scale_w = resize_pair_im(im, 250, 375)
    height, width = im_resize.shape[:2]
    # print(height)
    # print(width)
    # im_info = np.array([height, width, im_scale], dtype=np.float32)
    # im_tensor = transform(im_resize, mean, std)


    # gt boxes: (x1, y1, x2, y2, cls, grpid, subgrpid)
    # pair box images
    if roi_rec['gt_classes'].size > 0:
        gt_inds = np.where(roi_rec['gt_classes'] != 0)[0]

        #adjust the postive pair and negative pair
        pos_neg_list = [0,0,1,1]
        pos_neg = random.choice(pos_neg_list)
        # print(gt_inds[-1])

        if pos_neg == 0:
            gt_index = random.sample(list(gt_inds), 2)

            # gt_box_1
            gt_box_1 = np.empty(6, dtype=np.float32)
            gt_box_1[0:4] = roi_rec['boxes'][gt_index[0]]

            grpnum = roi_rec['objs'][gt_index[0]]
            grpids = grpnum['groupnum']
            grpids_list = grpids.split('_')
            grpid, subgrpid = int(grpids_list[0]), int(grpids_list[1])
            gt_box_1[4] = grpid
            gt_box_1[5] = subgrpid

            #gt_box_2
            gt_box_2 = np.empty(6, dtype=np.float32)
            gt_box_2[0:4] = roi_rec['boxes'][gt_index[1]]

            grpnum = roi_rec['objs'][gt_index[1]]
            grpids = grpnum['groupnum']
            grpids_list = grpids.split('_')
            grpid, subgrpid = int(grpids_list[0]), int(grpids_list[1])
            gt_box_2[4] = grpid
            gt_box_2[5] = subgrpid

        if pos_neg == 1:
            gt_index = random.sample(list(gt_inds),1)

            #gt_box_1
            gt_box_1 = np.empty(6, dtype=np.float32)
            gt_box_1[0:4] = roi_rec['boxes'][gt_index[0]]

            grpnum = roi_rec['objs'][gt_index[0]]
            grpids = grpnum['groupnum']
            grpids_list = grpids.split('_')
            grpid, subgrpid = int(grpids_list[0]), int(grpids_list[1])
            gt_box_1[4] = grpid
            gt_box_1[5] = subgrpid

            if gt_index[0] != gt_inds[-1]:
                gt_index_bef, gt_index_aft = gt_index[0]-1, gt_index[0]+1
                grpnum_bef = roi_rec['objs'][gt_index_bef]
                grpids_bef = grpnum_bef['groupnum']
                grpnum_aft = roi_rec['objs'][gt_index_aft]
                grpids_aft = grpnum_aft['groupnum']
                if grpids==grpids_bef:
                    gt_index.append(gt_index_bef)

                if grpids != grpids_bef and grpids==grpids_aft:
                    gt_index.append(gt_index_aft)

                if grpids != grpids_bef and grpids!=grpids_aft:
                    gt_index.append(gt_index_bef)

            if gt_index[0] == gt_inds[-1]:
                gt_index.append(gt_index[0]-1)

            gt_box_2 = np.empty(6, dtype=np.float32)
            gt_box_2[0:4] = roi_rec['boxes'][gt_index[1]]

            grpnum = roi_rec['objs'][gt_index[1]]
            grpids = grpnum['groupnum']
            grpids = grpids.split('_')
            grpid, subgrpid = int(grpids[0]), int(grpids[1])
            gt_box_2[4] = grpid
            gt_box_2[5] = subgrpid

    else:
        gt_box_1 = np.empty(6, dtype=np.float32)
        gt_box_2 = np.empty(6, dtype=np.float32)

    imwrite_path = './testmask/'+ roi_rec['index'] + '_'+ str(int(gt_box_1[4])) + '_'+ str(int(gt_box_1[5])) + '.jpg'
    x, y, x_max, y_max = gt_box_1[0:4]
    pair_im_1 = im[int(y):int(y_max),int(x):int(x_max)]
    pair_im_1,_,_ = resize_pair_im(pair_im_1, 250, 100)
    # cv2.imwrite(imwrite_path,pair_im_1)
    pair_im_1_tensor = transform_resnet(pair_im_1, mean, std)

    imwrite_path_2 = './testmask/' + roi_rec['index'] + '_' + str(int(gt_box_2[4]) )+ '_' + str(int(gt_box_2[5])) + '_a.jpg'
    x, y, x_max, y_max = gt_box_2[0:4]
    pair_im_2 = im[int(y):int(y_max), int(x):int(x_max)]
    pair_im_2,_,_ = resize_pair_im(pair_im_2, 250, 100)
    # cv2.imwrite(imwrite_path_2,pair_im_2)
    pair_im_2_tensor = transform_resnet(pair_im_2, mean, std)

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
    path = "./testmask/" + roi_rec["index"] + "_lo.jpg"
    # cv2.imwrite(path,location_map_1)
    location_map_1_tensor = transform_map(location_map_1)
    location_map_2 = location_map_im(im_resize.shape, gt_box_2)
    path = "./testmask/" + roi_rec["index"] + "_lo2.jpg"
    # cv2.imwrite(path,location_map_2)
    location_map_2_tensor = transform_map(location_map_2)

    # head size and head map
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
    path = "./testmask/" + roi_rec["index"] + ".jpg"
    # cv2.imwrite(path,r_map)
    r_map_tensor = transform_map(r_map)

    headsize_1 = np.array([(ymax1-y1), (xmax1-x1)])
    # print(headsize_1)
    headsize_2 = np.array([(ymax2-y2), (xmax2-x2)])
    # print(headsize_2)

    return pair_im_1_tensor, pair_im_2_tensor, location_map_1_tensor,location_map_2_tensor, \
           gt_box_1, gt_box_2,r_map_tensor,headsize_1,headsize_2


def imdecode(image_path):
    """Return BGR image read by opencv"""
    import os
    assert os.path.exists(image_path), image_path + ' not found'
    im = cv2.imread(image_path)
    return im

def resize(im, short, max_size):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param short: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :return: resized image (NDArray) and scale (float)
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(short) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    return im, im_scale

def resize_pair_im(im, resize_h, resize_w):
    im_size_h = im.shape[0]
    im_size_w = im.shape[1]
    im_scale_h = float(resize_h) / float(im_size_h)
    im_scale_w = float(resize_w) / float(im_size_w)
    im = cv2.resize(im, None, None, fx=im_scale_w, fy=im_scale_h, interpolation=cv2.INTER_LINEAR)
    return im, im_scale_h, im_scale_w

def transform_resnet(im, mean, std):
    """
    transform into mxnet tensor,
    subtract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param mean: [RGB pixel mean]
    :param std: [RGB pixel std var]
    :return: [batch, channel, height, width]
    """
    im = mx.nd.array(im.astype(np.float32))
    im = im / 255
    im = mx.image.color_normalize(im,mx.nd.array(mean),mx.nd.array(std))
    im = mx.nd.transpose(im,(2,0,1))
    im = im.asnumpy()
    '''
    im_tensor = np.zeros((3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[i, :, :] = (im[:, :, 2 - i] - mean[i]) / std[i]
    '''
    return im

def transform_map(im):
    """
    transform into mxnet tensor,
    subtract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param mean: [RGB pixel mean]
    :param std: [RGB pixel std var]
    :return: [batch, channel, height, width]
    """
    im = mx.nd.array(im.astype(np.float32))
    # im = im / 255
    im = mx.image.color_normalize(im,mx.nd.array([0,0,0]),mx.nd.array([1,1,1]))
    im = mx.nd.transpose(im,(2,0,1))
    im = im.asnumpy()
    '''
    im_tensor = np.zeros((3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[i, :, :] = (im[:, :, 2 - i] - mean[i]) / std[i]
    '''
    return im

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


def location_map_heads_ori(im_shape, head1,head2):

    im = np.zeros([im_shape[0], im_shape[1], 3], dtype=np.uint8)

    hx, hy, hxmax, hymax = head1[0:4]
    h = hymax -hy
    w = hxmax - hx

    if hx < 0: hx = 1
    if hy < 0: hy = 1
    if hxmax > im_shape[1]: hxmax = im_shape[1] - 1
    if hymax > im_shape[0]: hymax = im_shape[0] - 1

    if im[hy:hymax, hx:hxmax, :].shape[0] != h:
        h = im[hy:hymax, hx:hxmax, :].shape[0]
    if im[hy:hymax, hx:hxmax, :].shape[1] != w:
        w = im[hy:hymax, hx:hxmax, :].shape[1]
    im[hy:hymax, hx:hxmax, :] = np.ones([h, w, 3])*255


    hx1, hy1, hxmax1, hymax1 = head2[0:4]
    h1 = hymax1 - hy1
    w1 = hxmax1 - hx1

    if hx1 < 0: hx1 = 1
    if hy1 < 0: hy1 = 1
    if hxmax1 > im_shape[1] : hxmax1  = im_shape[1] - 1
    if hymax1 > im_shape[0] : hymax1 = im_shape[0] - 1

    if im[hy1:hymax1, hx1:hxmax1, :].shape[0] != h1:
        h1 = im[hy1:hymax1, hx1:hxmax1, :].shape[0]
    if im[hy1:hymax1, hx1:hxmax1, :].shape[1] != w1:
        w1 = im[hy1:hymax1, hx1:hxmax1, :].shape[1]
    im[hy1:hymax1, hx1:hxmax1, :] = np.ones([h1, w1, 3])*255
    # im = cv2.GaussianBlur(im,(5,5),0)
    im = cv2.GaussianBlur(im,(5,5),0)
    return im

def transform_inverse(im_tensor, mean, std):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param mean: [RGB pixel mean]
    :param std: [RGB pixel std var]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 3
    im = im_tensor.transpose((1, 2, 0))
    im = im * std + mean
    im = im.astype(np.uint8)
    return im


def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors by adding a new axis
    expand dims if only 1 tensor
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    if len(tensor_list) == 1:
        return tensor_list[0][np.newaxis, :]

    ndim = len(tensor_list[0].shape)
    dimensions = [len(tensor_list)]  # first dim is batch size
    for dim in range(ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))

    dtype = tensor_list[0].dtype
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind, :tensor.shape[0]] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind, :tensor.shape[0], :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind, :tensor.shape[0], :tensor.shape[1], :tensor.shape[2]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor
