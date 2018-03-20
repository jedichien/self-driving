from __future__ import print_function
from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.backend as K
from keras.optimizers import Adam
import os

from model import E2EModel
from process_data import load_dataset_df, pd2np, split_train_test, generate_batch, moderate_dataset

from config import config

def train_process(train, test, epochs=1, steps_per_epoch=100, validation_steps=100, lr=1e-3):
    optimizer = Adam(lr=lr)
    model = E2EModel()
    model.compile(loss='mse', optimizer=optimizer)
    
    if not os.path.exists('weights'):
        os.makedirs('weights')
    checkpoint_callback = ModelCheckpoint(os.path.join('weights', 'w.{epoch:02d}-{val_loss:.5f}.hd5'))
    logger = CSVLogger(filename='history.csv')

    model.fit_generator(generator=generate_batch(train, batch_size=config['batch_size'], bias=config['bias']),
                        steps_per_epoch=steps_per_epoch, 
                        epochs=epochs,
                        validation_data=generate_batch(test, batch_size=config['batch_size'], bias=1.0),
                        validation_steps=validation_steps,
                        callbacks=[checkpoint_callback, logger])
    
if __name__ == '__main__':
    dpath = os.path.join('data')
    print("Loading dataset: {}".format(dpath))
    ddf = load_dataset_df(dpath)
    print("Moderate dataset")
    ddf = moderate_dataset(ddf)
    print("DataFrame to numpy array")
    data = pd2np(ddf)
    print("Dataset splitted")
    train, test = split_train_test(data)
    print("Train: {}\nTest: {}".format(train.shape[0], test.shape[0]))
    print("Start to training")
    train_process(train, test, epochs=500, steps_per_epoch=5*config['batch_size'], validation_steps=config['batch_size'])
    
