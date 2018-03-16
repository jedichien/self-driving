from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas as pd
import os
import random
from random import shuffle
from config import w, h, c, config


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
"""
def draw_histogram_of_steering_angle(ddf, save=True):
    f, ax = plt.subplots(1, 1)
    ax.set_title('Steering Angle Distribution in Fronted Camera')
    fhist = ax.hist(ddf['steer'].values, 100, density=False, facecolor='green', alpha=0.7, width=0.03)
    ax.set_ylabel('frames')
    ax.set_xlabel('steering angle')
    # image saved
    if save:
        if not os.path.exists('output'):
            os.makedirs('output')
        plt.savefig(os.path.join('output', 'steering_angle.jpg'))
    plt.show()
    
"""
Pandas dataframe to numpy.matrix so called numpy.array.

Args:
  ddf: pandas dataframe
Return:
  numpy.matrix 
"""
def pd2np(ddf):
    return ddf.as_matrix()

"""
Split dataset into training and testing according to specific ratio rate.

Args:
  data: dataset
  ratio: ratio rate for training part.
Return:
  train and test dataset
"""
def split_train_test(data, ratio=0.8):
    # shuffle
    shuffle(data)
    size = int(data.shape[0]*ratio)
    train = data[0:size]
    test = data[size:]
    return train, test

"""
Data preprocessing including cropped and resized.

Args:
  imgp: path of image
  verbose: verbose if True
  save: saving image if True
Return:
  processed data
"""
def preprocessing_data(imgp, verbose=False, save=True):
    img = mpimg.imread(imgp)
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
        shuffle(data)
        n_current = 0
        X = np.zeros(shape=(batch_size, h, w, c), dtype=np.uint8)
        y_steer = np.zeros(shape=(batch_size,), dtype=np.float32)
        
        for d in data:
            if n_current == batch_size:
                break
            # choose frame randomly among (front, left, right)
            cameraid = random.randint(0, 2)
            frame = preprocessing_data(d[cameraid])
            if cameraid == 0:
                steer = d[3]
            elif cameraid == 1:
                steer = d[3] + config['delta_correction']
            else:
                steer = d[3] - config['delta_correction']

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
            if (abs(steer) + bias) < steer_magnitude_thresh:
                pass
            else:
                X[n_current] = frame
                y_steer[n_current] = steer
                n_current += 1
                
        yield X, y_steer
        
        