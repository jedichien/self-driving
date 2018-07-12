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
data histogram from sample of steering angles.

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
In order to preventing the effection by most frequency, zero steering
, I decrease it to secondly most frequency size.

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
Transform Pandas dataframe to numpy.matrix so called numpy.array.

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
                    float(df['speed'])])
    imgs = np.array(imgs)
    labels = np.array(labels, dtype=np.float32)
    
    return imgs, labels

"""
Split dataset into training and testing set according to specific ratio rate.

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
Need to augment data or not

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
def augmentShift(frame, cam_shift, verbose=False, save=False):
    rows, cols, _ = frame.shape
    shifted_point = [cols / 2 + cam_shift, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], shifted_point])
    M = cv2.getAffineTransform(pts1, pts2)
    _frame = np.copy(frame)
    _frame = cv2.warpAffine(_frame, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    if verbose:
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].set_title('Origin')
        ax[0].imshow(np.uint8(frame))
        ax[1].set_title('Shifted')
        ax[1].imshow(np.uint8(_frame))
        if save:
            if not os.path.exists('output'):
                os.makedirs('output')
            plt.savefig(os.path.join('output', 'shift.jpg'))
        plt.show()
    return _frame
"""
Rotate frame

Args:
  frame: input frame
  cam_degree: degree of rotation
Args:
  np.array
"""
def augmentRotate(frame, cam_degree, verbose=False, save=False):
    rows, cols, _ = frame.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), cam_degree, 1.0)
    _frame = cv2.warpAffine(frame, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    if verbose:
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].set_title('Origin')
        ax[0].imshow(np.uint8(frame))
        ax[1].set_title('Rotated')
        ax[1].imshow(np.uint8(_frame))
        if save:
            if not os.path.exists('output'):
                os.makedirs('output')
            plt.savefig(os.path.join('output', 'rotated.jpg'))
        plt.show()
    return _frame

def augmentBrightness(frame, jiter_mask, verbose=False, save=False):
    _frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    _frame[:, :, 2] *= jiter_mask
    _frame[:, :, 2] = np.clip(_frame[:, :, 2], a_min=0, a_max=255)
    _frame = cv2.cvtColor(_frame, cv2.COLOR_HSV2RGB)
    if verbose:
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].set_title('Origin')
        ax[0].imshow(np.uint8(frame))
        ax[1].set_title('Brightness')
        ax[1].imshow(np.uint8(_frame))
        if save:
            if not os.path.exists('output'):
                os.makedirs('output')
            plt.savefig(os.path.join('output', 'brightness.jpg'))
        plt.show()
    return _frame

def augmentHorizontalFlipped(frame, verbose=False, save=False):
    _frame = frame[:, ::-1, :]
    if verbose:
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].set_title('Origin')
        ax[0].imshow(np.uint8(frame))
        ax[1].set_title('Flipped')
        ax[1].imshow(np.uint8(_frame))
        if save:
            if not os.path.exists('output'):
                os.makedirs('output')
            plt.savefig(os.path.join('output', 'flipped.jpg'))
        plt.show()
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
        y_speed = np.zeros(shape=(batch_size,), dtype=np.float32)
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
            # max speed 
            speed = np.clip(((label[-1] - config['max_speed']/2)/config['max_speed']/2)-1, a_min=-1.0, a_max=1.0)
            
            if augmented:
                # camera shifting or rotating
                if needAugmented(augmented):
                    cam_shift = np.random.uniform(-100, 100)
                    frame = augmentShift(frame, cam_shift)
                    steer = steer + (cam_shift/(frame.shape[0]/2) * 180/(np.pi*25.0)/6.0)
                else:
                    cam_degree = np.random.uniform(-10, 10)
                    frame = augmentRotate(frame, cam_degree)
                    steer = steer - np.sin(cam_degree*np.pi/180)
                # horizontal flipping
                if random.choice([True, False]):
                    frame = augmentHorizontalFlipped(frame)
                    steer *= -1.
                # perturb slightly steering angle
                steer += np.random.normal(loc=0, scale=config['augmentation_steer_sigma'])
                # if colorful image randomly change brightness
                if config['input_channels'] == 3:
                    jiter_mask = random.uniform(config['augmentation_value_min'], config['augmentation_value_max'])
                    frame = augmentBrightness(frame, jiter_mask, verbose=False, save=False)
            # whether steer angle matchs the condition
            steer_magnitude_thresh = np.random.rand()
            if ((abs(steer) + bias) < steer_magnitude_thresh) or (steer < -1. and steer > 1.):
                pass
            else:
                X[n_current] = frame
                y_steer[n_current] = steer
                y_speed[n_current] = speed 
                n_current += 1
            if n_current == batch_size:     
                break
        yield X, zip(y_steer,y_speed)
