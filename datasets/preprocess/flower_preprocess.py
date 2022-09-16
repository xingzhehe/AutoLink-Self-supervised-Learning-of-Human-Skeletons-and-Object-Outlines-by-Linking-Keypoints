import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import h5py
import scipy.io as sio

img_path = '../../data/flower_raw/jpg'

target_size = 128

split = sio.loadmat('../../data/flower_raw/setid.mat')

print(split.keys())
print(len(split['trnid'][0]), len(split['valid'][0]), len(split['tstid'][0]))
print(split['tstid'])

train_img = []
test_img = []

for idx in split['tstid'][0]:
    img_file_name = 'image_{}.jpg'.format(str(idx).zfill(5))

    img = Image.open(os.path.join(img_path, img_file_name)).resize((target_size, target_size), resample=Image.BILINEAR)

    img = np.asarray(img)

    img = img.transpose((2, 0, 1)).reshape((3, target_size, target_size))
    train_img.append(img)


for idx in split['trnid'][0]:
    img_file_name = 'image_{}.jpg'.format(str(idx).zfill(5))

    img = Image.open(os.path.join(img_path, img_file_name)).resize((target_size, target_size), resample=Image.BILINEAR)

    img = np.asarray(img)

    img = img.transpose((2, 0, 1)).reshape((3, target_size, target_size))
    test_img.append(img)

print(len(train_img), len(test_img))

file = h5py.File('../../data/flower/flower.h5', "w")
file.create_dataset("train_img", np.shape(train_img), h5py.h5t.STD_U8BE, data=train_img)
file.create_dataset("test_img", np.shape(test_img), h5py.h5t.STD_U8BE, data=test_img)
file.close()
