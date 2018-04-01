from __future__ import division

import json
import numpy as np
import cv2
import tensorflow as tf
import io
import os
from PIL import Image
from utils import get_heat_maps, get_vec_maps

size = 368
if not os.path.exists('records'):
    os.mkdir('records')

rootdir = ['ai_challenger_keypoint_train_20170909', 'ai_challenger_keypoint_validation_20170911']
jsonfile = ['keypoint_train_annotations_20170909.json', 'keypoint_validation_annotations_20170911.json']
imagedir = ['keypoint_train_images_20170902', 'keypoint_validation_images_20170911']
filenames = ['records/train{}.records', 'records/val{}.records']


def resize(img, size):
    results = np.ones((size, size, 3), dtype=np.uint8)*128
    h,w,_ = img.shape
    scale = size/h if h>w else size/w
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    h, w, _ = img.shape
    results[:h, :w, :] = img
    return results, scale

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mid_1=[1, 2, 1, 7, 8, 1, 14, 14, 4, 5, 4, 10, 11]
mid_2=[2, 3, 7, 8, 9, 14, 13, 4, 5, 6, 10, 11, 12]

for tt in [0,1]:
    annos = json.load(open(os.path.join(rootdir[tt], jsonfile[tt]), 'r'))
    filename = filenames[tt]
    root = os.path.join(rootdir[tt], imagedir[tt])
    jj = 1
    writer = tf.python_io.TFRecordWriter(filename.format(jj))
    print('Writing', filename.format(jj))
    for index, anno in enumerate(annos):
        raw_image = cv2.imread(os.path.join(root, anno['image_id']+'.jpg'))
        raw_image = raw_image[:,:,::-1]
        raw_image, scale = resize(raw_image, size)
        # im = Image.fromarray(raw_image)
        # encoded_jpg = io.BytesIO()
        # im.save(encoded_jpg, 'JPEG')
        # encoded_jpg = encoded_jpg.getvalue()
        h, w, _ = raw_image.shape
        mask = np.zeros((h, w), dtype=np.float32)
        for k in anno['human_annotations']:
            x0,y0,x1,y1=anno['human_annotations'][k]
            x0,y0,x1,y1 = x0*scale,y0*scale,x1*scale,y1*scale
            x0,y0,x1,y1 = map(round, (x0,y0,x1,y1))
            x0,y0,x1,y1 = map(int, (x0,y0,x1,y1))
            mask[y0:y1,x0:x1] = 1
        num_person = len(anno['human_annotations'])
        heat_label = np.zeros((size//8,size//8,14), dtype=np.float32)
        vec_label = np.zeros((size//8,size//8,26), dtype=np.float32)
        kps = anno['keypoint_annotations']
        for n in range(14):
            heatn = np.zeros((size//8,size//8,num_person))
            for k, v in enumerate(kps):
                visible = kps[v][n*3+2]
                x = kps[v][n*3]
                y = kps[v][n*3+1]
                x *= scale
                y *= scale
                heatn[:,:,k] = get_heat_maps(x, y, visible, 8, size//8, size//8, 7.0)
            heat_label[:,:,n] = np.max(heatn, axis=-1)
        for n in range(13):
            vecn = np.zeros((size//8,size//8,2), dtype=np.float32)
            count = np.ones((size//8,size//8), dtype=np.float32)
            for k, v in enumerate(kps):
                kp1 = mid_1[n]-1
                kp2 = mid_2[n]-1
                visible1 = kps[v][kp1*3+2]
                visible2 = kps[v][kp2*3+2]
                if visible1==3 or visible2==3:
                    continue
                x1 = kps[v][kp1*3]
                y1 = kps[v][kp1*3+1]
                x2 = kps[v][kp2*3]
                y2 = kps[v][kp2*3+1]
                x1 *= scale
                y1 *= scale
                x2 *= scale
                y2 *= scale
                vecn[:,:,:] += get_vec_maps(x1, y1, x2, y2, visible1, visible2, size//8, size//8, 1.0, count)
            vec_label[:,:,2*n:2*n+2] = vecn / count[..., np.newaxis]
        mask = cv2.resize(mask, (size//8,size//8), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.float32)
        mask_raw = mask.tostring()
        heat_raw = heat_label.tostring()
        vec_raw = vec_label.tostring()
        raw_image = raw_image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
                'mask_raw': _bytes_feature(mask_raw),
                'heat_raw': _bytes_feature(heat_raw),
                'vec_raw': _bytes_feature(vec_raw),
                'image_raw': _bytes_feature(raw_image)}))
        writer.write(example.SerializeToString())
        if (index+1) % 500==0 and (index+1)!=len(annos):
            writer.close()
            jj +=1
            writer = tf.python_io.TFRecordWriter(filename.format(jj))
            print('Writing', filename.format(jj))
    writer.close()