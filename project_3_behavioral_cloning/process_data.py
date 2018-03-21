from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas as pd
import os
import random
from config import w, h, c, config
from sklearn.utils import shuffle

"""
Load raw dataset gotten from Udacity simulator.

Args:
  dpath: dataset directory
Return:
  pandas dataframe with parsed data.
"""
def load_dataset_df(dpath):
    head_csv = ['front', 'left', 'right', 'steer', 'throttle', 'brake', 'speed']
    csv_df = pd.read_csv(os.path.join(dpath, 'driving_log.csv'), names=head_csv)
    csv_df['front'] = [os.path.join(dpath, 'IMG', front.split('\\')[-1]) for front in csv_df['front']]
    csv_df['left'] = [os.path.join(dpath, 'IMG', front.split('\\')[-1]) for front in csv_df['left']]
    csv_df['right'] = [os.path.join(dpath, 'IMG', front.split('\\')[-1]) for front in csv_df['right']]
    return csv_df

"""
Data a histogram of sample of steering angles.

Args:
  ddf: pandas dataframe
  save: saving image if True
  filename: saving filename
"""
def draw_histogram_of_steering_angle(ddf, save=True, filename='steering_angle.jpg'):
    f, ax = plt.subplots(1, 1)
    ax.set_title('Steering Angle Distribution in Fronted Camera')
    fhist = ax.hist(ddf['steer'].values, 100, density=False, facecolor='green', alpha=0.7, width=0.03)
    ax.set_ylabel('frames')
    ax.set_xlabel('steering angle')
    # image saved
    if save:
        if not os.path.exists('output'):
            os.makedirs('output')
        plt.savefig(os.path.join('output', filename))
    plt.show()

"""
In order to preventing the effection by most frequency, zero steering. 
I reduce it to size of second frequency.

Args:
  ddf: pandas dataframe
Return:
  pandas dataframe
"""
def moderate_dataset(ddf):
    count, divs = np.histogram(ddf['steer'], bins=100)
    idx = count.argsort()[::-1][:2]

    f_c = count[idx[0]]
    s_c = count[idx[1]]
    f_v = divs[idx[0]]

    drop_idx = ddf['steer'] == f_v
    drop_idx = ddf['steer'][drop_idx].index
    drop_idx = random.sample(drop_idx, f_c - s_c)
    ddf = ddf.drop(drop_idx)
    return ddf    
    
"""
Pandas dataframe to numpy.matrix so called numpy.array.

Args:
  ddf: pandas dataframe
Return:
  numpy.array
"""
def pd2np(ddf):
    imgs = []
    labels = []
    
    for id, df in ddf.iterrows():
        imgs.append([df['front'],
                     df['left'],
                     df['right']])

        labels.append([float(df['steer']),
                    df['throttle'],
                    df['brake'],
                    df['speed']])
    imgs = np.array(imgs)
    labels = np.array(labels, dtype=np.float32)
    
    return imgs, labels

"""
Split dataset into training and testing according to specific ratio rate.

Args:
  data: dataset
  ratio: ratio rate for training part.
Return:
  train and test dataset
"""
def split_train_test(imgs, labels, ratio=0.8):
    # shuffle
    _imgs, _labels = shuffle(imgs, labels)
    size = int(_labels.shape[0]*ratio)
    train = (_imgs[:size], _labels[:size])
    test = (_imgs[size:], _labels[size:])
    return train, test

"""
Data preprocessing including cropped and resized.

Args:
  img: path of image or image array only if train is True
  train: default is False.
  verbose: verbose if True
  save: saving image if True
Return:
  processed data
"""
def preprocessing_data(img, train=False, verbose=False, save=True):
    if train:
        img = mpimg.imread(img)
    cropped = img[config['crop_height'], :, :]
    resized = cv2.resize(cropped, dsize=(w, h))
    # if input_channel equals to 1, changed color space from RGB to YUV
    if config['input_channels'] == 1:
        resized = np.expand_dims(cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)[:, :, 0], 2)
    
    if verbose:
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].set_title('Origin')
        ax[0].imshow(img)
        ax[1].set_title('Processed')
        ax[1].imshow(resized)
        if save:
            if not os.path.exists('output'):
                os.makedirs('output')
            plt.savefig(os.path.join('output', 'preprocessing.jpg'))
        plt.show()
        
    return resized.astype(np.float32)

"""
Need data augmented or not

Args:
  enable: use normal dist as threshold for condition to choose it;
  On the otherhand allways return False
"""
def needAugmented(enable):
    return np.random.uniform() > 0.5 if enable else False

"""
Shifting frame
positive means right, however negative means left
Args:
  frame: input frame
  cam_shift: shift distance
Return:
  np.array
"""
def augmentShift(frame, cam_shift):
    rows, cols, _ = frame.shape
    p = -cam_shift * 72
    pts1 = np.float32([[.0, .0], [300., .0], [.0, 100.]])
    pts2 = np.float32([[.0, .0], [300., .0], [p, .0]])
    M = cv2.getAffineTransform(pts1, pts2)
    _frame = np.copy(frame)
    _frame[60:, :, :] = cv2.warpAffine(_frame[60:, :], M, (cols, rows-60))

    return _frame
"""
Rotating frame

Args:
  frame: input frame
  cam_degree: degree of rotation
Args:
  np.array
"""
def augmentRotate(frame, cam_degree):
    p = -int(cam_degree*2)
    _frame = np.zeros(frame.shape, dtype=frame.dtype)
    
    if p == 0:
        return frame
    elif p > 0:
        _frame[:, p:, :] = frame[:, :-p, :]
    else:
        _frame[:, :p, :] = frame[:, -p:, :]

    return _frame

"""
Generate a batch of data.

Args:
  data: data for generating
  batch_size: size of batching
  augmented: Data augmentation if True
  bias: bias for checking steering angles condition
Return:
  training/testing dataset
"""
def generate_batch(data, batch_size=128, augmented=True, bias=0.5):
    while True:
        imgs, labels = data
        imgs, labels = shuffle(imgs, labels)
        n_current = 0
        X = np.zeros(shape=(batch_size, h, w, c), dtype=np.float32)
        y_steer = np.zeros(shape=(batch_size,), dtype=np.float32)
        
        for idx in range(labels.shape[0]):
            img = np.copy(imgs[idx])
            label = np.copy(labels[idx])
            # choose frame randomly among (front, left, right)
            cameraid = random.randint(0, 2)
            frame = preprocessing_data(img[cameraid], train=True)
            if cameraid == 0:
                steer = label[0]
            elif cameraid == 1:
                steer = label[0] + config['delta_correction']
            else:
                steer = label[0] - config['delta_correction']
           
            # camera shifting or rotating
            if needAugmented(augmented):
                if needAugmented(augmented):
                    cam_shift = np.random.uniform(-1, 1)
                    frame = augmentShift(frame, cam_shift)
                    steer = steer - cam_shift * 0.2 / 0.9
                else:
                    cam_degree = np.random.uniform(-10, 10)
                    frame = augmentRotate(frame, cam_degree)
                    steer = steer - 0.5 * cam_degree / 25.

            # camera flipped or colour space changed.
            if augmented:
                # horizontal flipping
                if random.choice([True, False]):
                    frame = frame[:, ::-1, :]
                    steer *= -1.
                
                # perturb slightly steering angle
                steer += np.random.normal(loc=0, scale=config['augmentation_steer_sigma'])
                
                # if colorful image randomly change brightness
                if config['input_channels'] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    frame[:, :, 2] *= random.uniform(config['augmentation_value_min'], config['augmentation_value_max'])
                    frame[:, :, 2] = np.clip(frame[:, :, 2], a_min=0, a_max=255)
                    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
                
            # whether steer angle matchs the condition
            steer_magnitude_thresh = np.random.rand()
            if ((abs(steer) + bias) < steer_magnitude_thresh) or (steer < -1. and steer > 1.):
                pass
            else:
                X[n_current] = frame
                y_steer[n_current] = steer
                n_current += 1
            if n_current == batch_size:     
                break
        yield X, y_steer
        
        
