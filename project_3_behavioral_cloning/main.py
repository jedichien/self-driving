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
    checkpoint_callback = ModelCheckpoint(os.path.join('weights', 'ep.{epoch:02d}-{val_loss:.5f}.pkl'))
    logger = CSVLogger(filename='history.csv')
    g_train = generate_batch(train, batch_size=config['batch_size'], bias=config['bias'])
    g_test = generate_batch(test, batch_size=config['batch_size'], bias=1.0, augmented=False)
    model.fit_generator(generator=g_train,
                        steps_per_epoch=steps_per_epoch, 
                        epochs=epochs,
                        validation_data=g_test,
                        validation_steps=validation_steps,
                        callbacks=[checkpoint_callback, logger])
    
if __name__ == '__main__':
    dpath = os.path.join('data')
    print("Loading dataset: {}".format(dpath))
    ddf = load_dataset_df(dpath)
    print("Moderate dataset")
    ddf = moderate_dataset(ddf)
    print("DataFrame to numpy array")
    imgs, labels = pd2np(ddf)
    print("Dataset splitted")
    train, test = split_train_test(imgs, labels)
    print("Train: {}\nTest: {}".format(train[1].shape[0], test[1].shape[0]))
    print("Start to training")
    train_process(train, test, epochs=500, steps_per_epoch=config['batch_size']//2, validation_steps=config['batch_size']//3)
    
