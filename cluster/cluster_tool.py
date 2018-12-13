import cv2
import numpy as np
import random
from sklearn import metrics

def check_affinity_matrix(affinity, n_cluster, threshold):
    num_single_list = []

    for i in range(affinity.shape[0]):
        num_count = 0
        for j in range(affinity.shape[1]):
            if affinity[i][j] < threshold:
                num_count += 1

        if num_count == affinity.shape[1]:
           # print affinity[i]
            #var = np.var(affinity[i])
            #print('var', var)
            num_single_list.append(i)

    n_cluster = n_cluster - len(num_single_list)

    if len(num_single_list) != 0:
        affinity = np.delete(affinity,num_single_list,axis=1)
        affinity = np.delete(affinity,num_single_list,axis=0)

    return affinity, n_cluster, num_single_list


def get_best_labels_f(sub_labels, label_length, single_list):
    final_best_labels = np.full(shape=label_length, fill_value=-1, dtype=int)

    cluster_number = np.max(sub_labels)
    cluster_single_num = cluster_number
    for ns in single_list:
        cluster_single_num += 1
        final_best_labels[ns] = cluster_single_num

    for sbl in range(len(sub_labels)):
        for bl in range(len(final_best_labels)):
            if final_best_labels[bl] == -1:
                final_best_labels[bl] = sub_labels[sbl]
                break

    return final_best_labels


def get_best_labels_t(sub_labels, cluster_number, label_length, single_list):
    final_best_labels = np.full(shape=label_length, fill_value=-1, dtype=int)

    cluster_single_num = cluster_number-1
    for ns in single_list:
        cluster_single_num += 1
        final_best_labels[ns] = cluster_single_num

    for sbl in range(len(sub_labels)):
        for bl in range(len(final_best_labels)):
            if final_best_labels[bl] == -1:
                final_best_labels[bl] = sub_labels[sbl]
                break

    return final_best_labels


def get_cluster_gt(roi_num, roidb):
    roi_batch = roidb[roi_num]

    objs = roi_batch['objs']
    clusters = 0
    label_gt = np.zeros(len(objs), dtype=int)
    for obj in range(len(objs)):
        if obj < len(objs)-1:
            g_1 = objs[obj]['groupnum']
            g_2 = objs[obj + 1]['groupnum']
            if g_1 == g_2:
                label_gt[obj] = clusters
                label_gt[obj+1] = clusters
            if g_1 != g_2:
                label_gt[obj] = clusters
                clusters += 1
                label_gt[obj+1] = clusters

    return label_gt


def pred_cluster_vis(label_pred, roidb, roi_num, out, pairwise_list):
    roi_batch = roidb[roi_num]
    input_file = './data/GroupActivityPerson/JPEGImages/' + roi_batch['index'] + '.jpg'
    im = cv2.imread(input_file)

    grouped_list=[]

    rois = roi_batch['objs']
    n_clusters = 1
    for i in range(len(label_pred)):
        if label_pred[i]+1 > n_clusters:
            n_clusters = label_pred[i]+1

    color_list = []
    for color_ind in range(n_clusters):
        color = (random.randint(30, 255)*1.2, random.randint(60, 255)*1.6, random.randint(60, 255)*1.4)
        color_list.append(color)

    for pl in pairwise_list:
        x, y, xmax, ymax = roi_batch['boxes'][pl[0]]
        x2, y2, xmax2, ymax2 = roi_batch['boxes'][pl[1]]
        center_gt_1_x = int(x) + int((xmax - x) / 2)
        center_gt_1_y = int(y) + int((ymax - y) / 4)
        center_gt_2_x = int(x2) + int((xmax2 - x2) / 2)
        center_gt_2_y = int(y2) + int((ymax2 - y2) / 4)

        cluster_num_1 = label_pred[pl[0]]
        cluster_num_2 = label_pred[pl[1]]

        if cluster_num_1 == cluster_num_2:
            cv2.circle(im, (center_gt_1_x, center_gt_1_y), 8, color_list[cluster_num_1], -1)
            cv2.circle(im, (center_gt_2_x, center_gt_2_y), 8, color_list[cluster_num_1], -1)
            cv2.line(im, (center_gt_1_x, center_gt_1_y), (center_gt_2_x, center_gt_2_y), color_list[cluster_num_1], 2, cv2.LINE_AA)
            if pl[0] not in grouped_list:
                grouped_list.append(pl[0])
            if pl[1] not in grouped_list:
                grouped_list.append((pl[1]))

    for ob in range(len(roi_batch['boxes'])):
        if ob not in grouped_list:
            x, y, xmax, ymax = roi_batch['boxes'][ob]
            cv2.rectangle(im, (int(x), int(y)), (int(xmax),int(ymax)), color_list[label_pred[ob]], 2)

    outpath = out + roi_batch['index'] + '_cls.jpg'
    grp_name = out + roi_batch['index'] + '_grp_gt.jpg'
    cv2.imwrite(outpath, im)

    print ('out image:{0}'.format(roi_batch['index']))


def group_clustering(c_affinity,roidb,index,n_cluster,pairwise_list,cut_thre,give_cluster,vis,path):

    affinity, cluster_num_t, single_list = check_affinity_matrix(c_affinity, n_cluster, cut_thre)
    roi_batch = roidb[index]
    label_length = len(roi_batch['objs'])
    # print(roi_batch['index'])

    if give_cluster == False:
        from cluster.spectral_cluster import SpectralClustering
        disort_list, labels_list, s_best_labels, var_dis = SpectralClustering(affinity='precomputed',
                                                                              n_clusters_range=affinity.shape[0] + 1,
                                                                              gamma=0.1, n_components=None).fit(affinity)
        s_best_labels = list(s_best_labels)

        gt = get_cluster_gt(index, roidb)
        best_labels = get_best_labels_f(s_best_labels, label_length, single_list)
        print(gt)
        print(best_labels)

        score = metrics.v_measure_score(gt, best_labels)
        print(score)

        outpath = path
        if vis == True:
            pred_cluster_vis(best_labels, roidb, index, outpath, pairwise_list)

    if give_cluster == True:
        from cluster.ori_spectral_cluster import SpectralClustering
        s_best_labels = SpectralClustering(affinity='precomputed', n_clusters=cluster_num_t, gamma=0.1).fit(affinity)
        s_best_labels = list(s_best_labels.labels_)

        gt = get_cluster_gt(index, roidb)
        best_labels = get_best_labels_t(s_best_labels, cluster_num_t, label_length, single_list)
        print(gt)
        print(best_labels)

        score = metrics.v_measure_score(gt, best_labels)
        print(score)

        outpath = path
        if vis == True:
            pred_cluster_vis(best_labels, roidb, index, outpath, pairwise_list)

    return gt, best_labels, score



