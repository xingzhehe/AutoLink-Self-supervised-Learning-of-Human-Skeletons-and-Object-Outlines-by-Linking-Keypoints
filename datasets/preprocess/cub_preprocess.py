import os

import h5py
import numpy as np
import scipy.io as sio
from PIL import Image
from matplotlib import colors


raw_path = '../../../data/cub_raw/CUB_200_2011'
new_path = '../../../data/cub'

target_size = 128

train_anno = sio.loadmat(os.path.join(raw_path, 'cachedir', 'cub', 'data', '{}_cub_cleaned.mat'.format('train')), struct_as_record=False, squeeze_me=True)['images']
val_anno = sio.loadmat(os.path.join(raw_path, 'cachedir', 'cub', 'data', '{}_cub_cleaned.mat'.format('val')), struct_as_record=False, squeeze_me=True)['images']
test_anno = sio.loadmat(os.path.join(raw_path, 'cachedir', 'cub', 'data', '{}_cub_cleaned.mat'.format('test')), struct_as_record=False, squeeze_me=True)['images']

test001_anno = [anno for anno in train_anno if anno.class_id == 1 and anno.test] + \
               [anno for anno in val_anno if anno.class_id == 1 and anno.test] + \
               [anno for anno in test_anno if anno.class_id == 1 and anno.test]
test002_anno = [anno for anno in train_anno if anno.class_id == 2 and anno.test] + \
               [anno for anno in val_anno if anno.class_id == 2 and anno.test] + \
               [anno for anno in test_anno if anno.class_id == 2 and anno.test]
test003_anno = [anno for anno in train_anno if anno.class_id == 3 and anno.test] + \
               [anno for anno in val_anno if anno.class_id == 3 and anno.test] + \
               [anno for anno in test_anno if anno.class_id == 3 and anno.test]

print(train_anno[0].__dict__.keys())


def clean_group_img_kp_vis(anno):
    group_img = []
    group_kp = []
    group_vis = []

    for i in range(len(anno)):
        rel_path = anno[i].rel_path
        bbox = anno[i].bbox
        kp = anno[i].parts[:2, :].transpose(-1, -2)
        visibility = anno[i].parts[2, :]
        img = Image.open(os.path.join(raw_path, 'images', rel_path))
        w, h = img.size

        # The following part is adapted from
        # https://github.com/subhc/unsup-parts/blob/3e30b372136517ef010e974503389a3ae6833bb2/evaluation/CUB%20eval.ipynb
        # calculate border width and height
        border_h = h % 16
        border_w = w % 16

        # calculate the new shape
        new_h = h - border_h
        new_w = w - border_w

        # cut the inputs
        img = img.crop((border_w // 2, border_h // 2, new_w + (border_w // 2), new_h + (border_h // 2)))

        for i in range(15):
            if visibility[i] == 1:
                kp[i][0] -= (border_w // 2)  # column shift
                kp[i][1] -= (border_h // 2)  # row shift

                # remove the landmarks if unlucky (never really stepped in for test set of cub200)
                if kp[i][0] < 0 or kp[i][0] >= new_w or kp[i][1] < 0 or kp[i][1] >= new_h:
                    kp[i][0] = 0
                    kp[i][1] = 0
                    visibility[i] = 0

        # transform the bounding box correspondingly

        box_x_center = (bbox.x1 + bbox.x2) / 2
        box_y_center = (bbox.y1 + bbox.y2) / 2

        bbox_w = bbox.x2 - bbox.x1
        bbox_h = bbox.y2 - bbox.y1
        bbox_w = max(bbox_w, bbox_h) / 2
        bbox_h = bbox_w

        bbox_x_min = max(0, box_x_center - bbox_w)
        bbox_x_max = min(w, box_x_center + bbox_w)
        bbox_y_min = max(0, box_y_center - bbox_h)
        bbox_y_max = min(h, box_y_center + bbox_h)

        bbox_x_min = max(0, bbox_x_min - (border_w // 2))
        bbox_x_max = min(new_w, bbox_x_max)
        bbox_y_min = max(0, bbox_y_min - (border_h // 2))
        bbox_y_max = min(new_h, bbox_y_max)

        single_bbox = (bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max)
        img = img.crop(single_bbox).resize((target_size, target_size), resample=Image.BILINEAR)

        bbox_w = bbox_x_max - bbox_x_min
        bbox_h = bbox_y_max - bbox_y_min
        # center coordinate space
        kp = np.concatenate(((kp[:, 1:2] - bbox_y_min) / bbox_h,
                             (kp[:, 0:1] - bbox_x_min) / bbox_w), axis=1).reshape(15, 2)

        img = np.asarray(img)
        try:
            img = img.transpose((2, 0, 1)).reshape((3, target_size, target_size))
        except:
            img = np.concatenate([img[None, :, :], img[None, :, :], img[None, :, :]], axis=0)

        group_img.append(img)
        group_kp.append(kp)
        group_vis.append(visibility)

    return group_img, group_kp, group_vis

anno_dict = {'train': train_anno, 'val': val_anno, 'test': test_anno,
             'test001': test001_anno, 'test002': test002_anno, 'test003': test003_anno}

group_dict = dict()
for split in ['train', 'val', 'test', 'test001', 'test002', 'test003']:
    anno = anno_dict[split]
    group_img, group_kp, group_vis = clean_group_img_kp_vis(anno)
    group_dict['{}_img'.format(split)] = group_img
    group_dict['{}_kp'.format(split)] = group_kp
    group_dict['{}_vis'.format(split)] = group_vis
    print(split, len(group_img))

file = h5py.File(os.path.join(new_path, 'cub.h5'), "w")
for key, value in group_dict.items():
    dtype = h5py.h5t.STD_U8BE if 'img' in key or 'vis' in key else "float32"
    file.create_dataset(key, np.shape(value), dtype, data=value)
file.close()
