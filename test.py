from __future__ import division

import argparse
import json
import os
import cv2
import math
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from inceptionv2 import inception_resnet_v2_arg_scope, inception_resnet_v2_base

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

slim = tf.contrib.slim

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[1, 2], [2, 3], [1, 7], [7, 8], [8, 9], [1, 14],
           [14, 13], [14, 4], [4, 5], [5, 6], [4, 10], [10, 11],
           [11, 12]]

# the middle joints heatmap correpondence
mapIdx = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], \
          [16, 17], [18, 19], [20, 21], [22, 23], [24, 25]]
# mapIdx = [[1, 0], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], \
#           [16, 17], [18, 19], [20, 21], [22, 23], [24, 25]]

output_stride = 8
stages = 6
np_branch1 = 26  # vec
np_branch2 = 14  # heat
is_training = False
weight_decay = 1e-5
size = 368
keyidx = 3
size__ = [0.5, 1, 1.5, 2]

def conv(x, nf, ks, name, weight_decay):
    x = tf.layers.batch_normalization(x, momentum=0.9997, training=is_training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, nf, ks, name=name, kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                         padding='same', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    return x


def stage1_block(x, num_p, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 3, "Mconv1_stage1_L%d" % branch, weight_decay)
    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, weight_decay)
    x = conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, weight_decay)
    x = conv(x, 512, 1, "Mconv4_stage1_L%d" % branch, weight_decay)
    x = conv(x, num_p, 1, "Mconv5_stage1_L%d" % branch, weight_decay)
    return x


def stageT_block(x, num_p, stage, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), weight_decay)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), weight_decay)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), weight_decay)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), weight_decay)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), weight_decay)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), weight_decay)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), weight_decay)
    return x


def get_resnet50():
    inputs = tf.placeholder(tf.float32, [None, None, None, 3])
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        stage0_out, end_points = resnet_v2.resnet_v2_50(inputs,
                                                        128,
                                                        is_training=False,
                                                        global_pool=False,
                                                        output_stride=output_stride)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
    net = tf.concat([stage1_branch1_out, stage1_branch2_out, stage0_out], -1)
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out = stageT_block(net, np_branch1, sn, 1, weight_decay)

        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out = stageT_block(net, np_branch2, sn, 2, weight_decay)

        net = tf.concat([stageT_branch1_out, stageT_branch2_out, stage0_out], -1)
    return stageT_branch1_out, stageT_branch2_out, inputs

def get_inceptionv2():
    # images are rescaled to -1~1 in resnet_v2
    inputs = tf.placeholder(tf.float32, [None, None, None, 3])
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        with tf.variable_scope('InceptionResnetV2', 'InceptionResnetV2', [inputs], reuse=None) as scope:
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                stage0_out, end_points = inception_resnet_v2_base(inputs, scope=scope)
    # feature map refine
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)

    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)

    net = tf.concat([stage1_branch1_out, stage1_branch2_out, stage0_out], -1)

    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out = stageT_block(net, np_branch1, sn, 1, weight_decay)

        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out = stageT_block(net, np_branch2, sn, 2, weight_decay)

        net = tf.concat(
            [stageT_branch1_out, stageT_branch2_out, stage0_out], -1)
    return stageT_branch1_out, stageT_branch2_out, inputs

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def process(input_image, vec, heat, inputs, sess):
    oriImg = cv2.imread(input_image)  # B,G,R order
    oriImg = oriImg[:,:,::-1]
    # plt.imshow(oriImg)
    # plt.show()
    multiplier = [x * size / oriImg.shape[0] for x in size__]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], np_branch2))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], np_branch1))

    for m in range(len(multiplier)):
        scale = multiplier[m]

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = padRightDownCorner(imageToTest, 8,
                                                     128)

        input_img = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]),
                                 (3, 0, 1, 2))  # required shape (1, width, height, channels)

        input_img /= 256
        input_img -= .5
        input_img *= 2
        paf, heatmap = sess.run([vec, heat], feed_dict={inputs: input_img})

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(heatmap)  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=8, fy=8,
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(paf)  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=8, fy=8,
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(np_branch2):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > 0.1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]-1]
        candB = all_peaks[limbSeq[k][1]-1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candA[i][:2], candB[j][:2])
                    # print(vec)
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    if norm==0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candB[j][0], candA[i][0], num=mid_num), \
                                        np.linspace(candB[j][1], candA[i][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > 0.05)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])
            connection_candidate = sorted(connection_candidate, key=lambda x: x[keyidx], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 16))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 14:
                    row = -1 * np.ones(16)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    persons = len(subset)

    return persons, subset, all_peaks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dir', type=str,
    #                     default='/media/xxx/Data/keypoint/data/ai_challenger_keypoint_test_a_20170923/keypoint_test_a_images_20170923/',
    #                     help='input image')
    # parser.add_argument('--dir', type=str,
    #                     default='/media/xxx/Data/keypoint/data/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911/',
    #                     help='input image')
    parser.add_argument('--dir', type=str,
                        default='../ai_challenger_keypoint_test_b_20171120/keypoint_test_b_images_20171120/',
                        help='input image')
    parser.add_argument('--output', type=str, default='result.json')
    parser.add_argument('--model', type=str, default='./saved_inception_resnetv2/model.ckpt-268932', help='path to the weights file')

    args = parser.parse_args()
    input_dir = args.dir
    output = args.output

    tic = time.time()
    print('start processing...')

    vec, heat, inputs = get_inceptionv2()
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, args.model)

    results = []

    # # generate image with body parts
    for k, v in enumerate(os.listdir(input_dir)):
        print("{}/{}".format(k+1, len(os.listdir(input_dir))))
        d = {}
        d["image_id"] = v.split('.')[0]
        d["keypoint_annotations"] = {}
        input_image = os.path.join(input_dir, v)
        persons, subset, all_peaks = process(input_image, vec, heat, inputs, sess)
        # print(persons, subset, all_peaks)
        flat_peaks = [item for sublist in all_peaks for item in sublist]
        for i in range(persons):
            d["keypoint_annotations"]["human{}".format(i+1)] = []
            for j in range(14):
                index = subset[i][j]
                if index==-1:
                    d["keypoint_annotations"]["human{}".format(i+1)].extend([0,0,0])
                else:
                    coord = flat_peaks[int(subset[i][j])]
                    d["keypoint_annotations"]["human{}".format(i+1)].extend([coord[0],coord[1],1])
        results.append(d)
    with open(output, 'w') as f:
        json.dump(results, f)

    # with open('part.json') as f:
    #     part = json.load(f)
    # # generate image with body parts
    # for k, v in enumerate(part):
    #     print("{}/{}".format(k+1, len(part)))
    #     d = {}
    #     d["image_id"] = v['image_id']
    #     d["keypoint_annotations"] = {}
    #     input_image = os.path.join(input_dir, v['image_id']+'.jpg')
    #     persons, subset, all_peaks = process(input_image, vec, heat, inputs, sess)
    #     # print(persons, subset, all_peaks)
    #     flat_peaks = [item for sublist in all_peaks for item in sublist]
    #     for i in range(persons):
    #         d["keypoint_annotations"]["human{}".format(i+1)] = []
    #         for j in range(14):
    #             index = subset[i][j]
    #             if index==-1:
    #                 d["keypoint_annotations"]["human{}".format(i+1)].extend([0,0,0])
    #             else:
    #                 coord = flat_peaks[int(subset[i][j])]
    #                 d["keypoint_annotations"]["human{}".format(i+1)].extend([coord[0],coord[1],1])
    #     results.append(d)
    # with open(output, 'w') as f:
    #     json.dump(results, f)

    toc = time.time()
    print('processing time is %.5f' % (toc - tic))
