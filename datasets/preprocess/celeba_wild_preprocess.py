import h5py
import numpy as np
from PIL import Image

target_size = 128
iou_threshold = 0.3

poses = []

landmarks = [l.split() for l in open('../../data/celeba_wild_raw/list_landmarks_celeba.txt') if len(l.split()) == 11]
landmarks_dict = {x[0]: np.array([(int(x[1]), int(x[2])), (int(x[3]), int(x[4])), (int(x[5]), int(x[6])), (int(x[7]), int(x[8])), (int(x[9]), int(x[10]))]) for
                  x in landmarks}

boxes = [l.split() for l in open('../../data/celeba_wild_raw/list_bbox_celeba.txt') if len(l.split()) == 5 and l[:8] != 'image_id']
box_dict = {x[0]: [int(x[1]), int(x[2]), int(x[3]), int(x[4])] for x in boxes}

train_id = [l.split()[0] for l in open('../../data/celeba_wild_raw/MAFL/celebA_training.txt')]
mafl_train_id = [l.split()[0] for l in open('../../data/celeba_wild_raw/MAFL/training.txt')]
mafl_test_id = [l.split()[0] for l in open('../../data/celeba_wild_raw/MAFL/testing.txt')]

train_img = []
train_landmark = []
for name in train_id:
    img_file = "../../data/celeba_wild_raw/img_celeba/{}".format(name)
    img = Image.open(img_file)
    h, w = img.size
    img = img.resize((target_size, target_size), resample=Image.BILINEAR)
    box = box_dict[name]
    
    if box[2]*box[3] < h*w*iou_threshold:
        continue

    lms = np.zeros((5, 2))
    lms[:, 0] = landmarks_dict[name][:, 1] * target_size / h
    lms[:, 1] = landmarks_dict[name][:, 0] * target_size / w

    # plt.imshow(img)
    # plt.scatter(lms[:, 1], lms[:, 0])
    # plt.show()

    train_img.append(np.asarray(img).transpose((2, 0, 1)).reshape((1, 3, target_size, target_size)))
    train_landmark.append(lms.reshape(1, 5, 2))

print('Training original {} filtered {}'.format(len(train_id), len(train_landmark)))

mafl_train_img = []
mafl_train_landmark = []
for name in mafl_train_id:
    img_file = "../../data/celeba_wild_raw/img_celeba/{}".format(name)
    img = Image.open(img_file)
    h, w = img.size
    img = img.resize((target_size, target_size), resample=Image.BILINEAR)
    box = box_dict[name]
    
    if box[2]*box[3] < h*w*iou_threshold:
        continue
        
    lms = np.zeros((5, 2))
    lms[:, 0] = landmarks_dict[name][:, 1] * target_size / h
    lms[:, 1] = landmarks_dict[name][:, 0] * target_size / w

    mafl_train_img.append(np.asarray(img).transpose((2, 0, 1)).reshape((1, 3, target_size, target_size)))
    mafl_train_landmark.append(lms.reshape(1, 5, 2))

print('MAFL Training original {} filtered {}'.format(len(mafl_train_id), len(mafl_train_landmark)))


mafl_test_img = []
mafl_test_landmark = []
for name in mafl_test_id:
    img_file = "../../data/celeba_wild_raw/img_celeba/{}".format(name)
    img = Image.open(img_file)
    h, w = img.size
    img = img.resize((target_size, target_size), resample=Image.BILINEAR)
    box = box_dict[name]
    
    if box[2]*box[3] < h*w*iou_threshold:
        continue
        
    lms = np.zeros((5, 2))
    lms[:, 0] = landmarks_dict[name][:, 1] * target_size / h
    lms[:, 1] = landmarks_dict[name][:, 0] * target_size / w

    mafl_test_img.append(np.asarray(img).transpose((2, 0, 1)).reshape((1, 3, target_size, target_size)))
    mafl_test_landmark.append(lms.reshape(1, 5, 2))

print('MAFL Test original {} filtered {}'.format(len(mafl_test_id), len(mafl_test_landmark)))

train_img = np.concatenate(train_img)
train_landmark = np.concatenate(train_landmark)
print(train_img.shape, train_landmark.shape)
mafl_train_img = np.concatenate(mafl_train_img)
mafl_train_landmark = np.concatenate(mafl_train_landmark)
mafl_test_img = np.concatenate(mafl_test_img)
mafl_test_landmark = np.concatenate(mafl_test_landmark)


file = h5py.File('../../data/celeba_wild/celeba_wild.h5', "w")
file.create_dataset("train_img", np.shape(train_img), h5py.h5t.STD_U8BE, data=train_img)
file.create_dataset("train_landmark", np.shape(train_landmark), "float32", data=train_landmark)
file.create_dataset("mafl_train_img", np.shape(mafl_train_img), h5py.h5t.STD_U8BE, data=mafl_train_img)
file.create_dataset("mafl_train_landmark", np.shape(mafl_train_landmark), "float32", data=mafl_train_landmark)
file.create_dataset("mafl_test_img", np.shape(mafl_test_img), h5py.h5t.STD_U8BE, data=mafl_test_img)
file.create_dataset("mafl_test_landmark", np.shape(mafl_test_landmark), "float32", data=mafl_test_landmark)
file.close()
