import cv2
import numpy as np


def get_head_img(yolo,roi):

    im = cv2.imread(roi['image'])
    headbox = yolo.predict(im)

    objs = roi["objs"]
    box_x_difference = 0
    box_y_difference = 0
    box_score = 0

    if len(headbox) != 0:
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
                if box.xmin < 0: box.xmin = 1
                if box.ymin < 0: box.ymin = 1
                if box.xmax > im.shape[1]: box.xmax = im.shape[1]
                if box.ymax > im.shape[0]: box.ymax = im.shape[0]

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
                    if box_xmin < x: box_xmin = x+1
                    box_ymin = (y + (ymax - y) / 25)
                    if box_ymin < y: box_ymin = y+1
                    head_list[i,:] = [i, box_xmin, box_ymin, box_xmin+box_l, box_ymin+box_l]
                if 1 <= ratio_im and ratio_im <= 2:
                    box_l = (ymax - y) / 4.5
                    box_xmin = center_im - (box_l / 2)
                    if box_xmin < x: box_xmin = x + 1
                    box_ymin = (y + (ymax - y) / 25)
                    if box_ymin < y: box_ymin = y + 1
                    head_list[i, :] = [i, box_xmin, box_ymin, box_xmin + box_l, box_ymin + box_l]
                if ratio_im < 1:
                    box_l = (ymax - y) / 3.5
                    box_xmin = center_im - (box_l / 2)
                    if box_xmin < x: box_xmin = x + 1
                    box_ymin = (y + (ymax - y) / 25)
                    if box_ymin < y: box_ymin = y + 1
                    head_list[i, :] = [i, box_xmin, box_ymin, box_xmin + box_l, box_ymin + box_l]


            bdbox = im[int(y):int(ymax), int(x):int(xmax)]
            bdbox_c = bdbox.copy()

            i_index,hxmin,hymin,hxmax,hymax = head_list[i,:]
            cv2.rectangle(bdbox_c, (int(hxmin-x),int(hymin-y)), (int(hxmax-x),int(hymax-y)), (0, 255, 0), 2)
            out_path_file = '/home/deeplearning/MO/soa/headsin/' + roi['index'] + str(x) + str(
                y) + '_detected.jpg'
            cv2.imwrite(out_path_file, bdbox_c)


    if len(headbox) == 0:
        head_list = np.zeros((len(objs), 5), dtype=np.uint16)
        for i in range(len(objs)):
            x, y, xmax, ymax = objs[i]['bbox']
            if x < 0: x = 1
            if xmax > im.shape[1]: xmax = im.shape[1]
            if y < 0: y = 1
            if ymax > im.shape[0]: ymax = im.shape[0]

            center_im = x + ((xmax - x) / 2)
            ratio_im = (ymax - y) / (xmax - x)
            if ratio_im > 2:
                box_l = (ymax - y) / 6.5
                box_xmin = center_im - (box_l / 2)
                if box_xmin < x: box_xmin = x + 1
                box_ymin = (y + (ymax - y) / 25)
                if box_ymin < y: box_ymin = y + 1
                head_list[i, :] = [i, box_xmin, box_ymin, box_xmin + box_l, box_ymin + box_l]
            if 1 <= ratio_im and ratio_im <= 2:
                box_l = (ymax - y) / 4.5
                box_xmin = center_im - (box_l / 2)
                if box_xmin < x: box_xmin = x + 1
                box_ymin = (y + (ymax - y) / 25)
                if box_ymin < y: box_ymin = y + 1
                head_list[i, :] = [i, box_xmin, box_ymin, box_xmin + box_l, box_ymin + box_l]
            if ratio_im < 1:
                box_l = (ymax - y) / 3.5
                box_xmin = center_im - (box_l / 2)
                if box_xmin < x: box_xmin = x + 1
                box_ymin = (y + (ymax - y) / 25)
                if box_ymin < y: box_ymin = y + 1
                head_list[i, :] = [i, box_xmin, box_ymin, box_xmin + box_l, box_ymin + box_l]

            bdbox = im[int(y):int(ymax), int(x):int(xmax)]
            bdbox_c = bdbox.copy()

            i_index,hxmin,hymin,hxmax,hymax = head_list[i,:]
            #cv2.rectangle(bdbox_c, (int(hxmin-x),int(hymin-y)), (int(hxmax-x),int(hymax-y)), (0, 255, 0), 2)
            out_path_file = './headsin/' + roi['index'] + str(x) + str(y) + '_detected.jpg'
            # cv2.imwrite(out_path_file, bdbox_c)

    roi_head_list = {'index': roi['index'],
                     'headbox': head_list}

    return roi_head_list
