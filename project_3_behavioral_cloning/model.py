from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, Flatten, ELU, Lambda, Dense
from config import h, w, c

def E2EModel(verbose=True):
    inputs = Input(shape=(h, w, c))
    # standardize input that applied here is to be accelerated via GPU processing.
    _ = Lambda(lambda c: c/127.5 - 1.)(inputs)
    
    _ = Conv2D(24, kernel_size=(5, 5), strides=(2, 2), padding='valid')(_)
    _ = ELU()(_)
    _ = Dropout(0.2)(_)
    
    _ = Conv2D(36, kernel_size=(5, 5), strides=(2, 2), padding='valid')(_)
    _ = ELU()(_)
    _ = Dropout(0.2)(_)
    
    _ = Conv2D(48, kernel_size=(5, 5), strides=(2, 2), padding='valid')(_)
    _ = ELU()(_)
    _ = Dropout(0.2)(_)
    
    _ = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid')(_)
    _ = ELU()(_)
    _ = Dropout(0.2)(_)
    
    _ = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid')(_)
    _ = ELU()(_)
    _ = Dropout(0.2)(_)

    _ = Flatten()(_)
    _ = Dense(100)(_)
    _ = ELU()(_)
    #_ = Dropout(0.5)(_)
    
    _ = Dense(50)(_)
    _ = ELU()(_)
    #_ = Dropout(0.5)(_)
    
    _ = Dense(10)(_)
    _ = ELU()(_)
    
    outputs = Dense(1)(_)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    if verbose:
        model.summary()
        
    return model
