import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from art.estimators.classification import KerasClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
import numpy as np
import keras
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import scipy

import time


# Define the CNN architecture
def create_aux_model(X_train, X_val, y_train, y_val, datagen, num_epochs, batch_size):
    
    # Define auxillary model

    keras.backend.clear_session()

    aux_model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding='same', 
                               activation='relu',
                               input_shape=(32, 32, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), padding='same', 
                            activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),
        
        keras.layers.Conv2D(64, (3, 3), padding='same', 
                               activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), padding='same',
                            activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),
        
        keras.layers.Conv2D(128, (3, 3), padding='same', 
                               activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), padding='same',
                            activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),
        
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.4),    
        keras.layers.Dense(10,  activation='softmax')
    ])
    
    lr_schedule = keras.callbacks.LearningRateScheduler(
                  lambda epoch: 1e-4 * 10**(epoch / 10))
    optimizer = keras.optimizers.Adam(lr=1e-4, amsgrad=True)
    aux_model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                     metrics=['accuracy'])

    aux_history = aux_model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
                              epochs=num_epochs, validation_data=(X_val, y_val),
                              callbacks=[lr_schedule])

    return aux_history


# Define actual model
def create_actual_model(X_train, X_val, y_train, y_val, datagen, num_epochs, batch_size):
    
    keras.backend.clear_session()
    
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding='same', 
                               activation='relu',
                               input_shape=(32, 32, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), padding='same', 
                            activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),
        
        keras.layers.Conv2D(64, (3, 3), padding='same', 
                               activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), padding='same',
                            activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),
        
        keras.layers.Conv2D(128, (3, 3), padding='same', 
                               activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), padding='same',
                            activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),
        
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.4),    
        keras.layers.Dense(10,  activation='softmax')
    ])
    
    early_stopping = keras.callbacks.EarlyStopping(patience=8)
    optimizer = keras.optimizers.Adam(lr=1e-3, amsgrad=True)
    model_checkpoint = keras.callbacks.ModelCheckpoint(
                       'C:/Users/USER/OneDrive - Universiti Teknikal Malaysia Melaka/Desktop/Paper 6/SVHN/best_cnn.h5', 
                       save_best_only=True)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history_actual = actual_model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
                              epochs=num_epochs, validation_data=(X_val, y_val),
                              callbacks=[early_stopping, model_checkpoint])

    return model, history_actual
    


def train(X_train, X_val, y_train, y_val, datagen, num_epochs, batch_size):
    t_start=time.time()
    # Fit model in order to determine best learning rate
    aux_model=create_aux_model(X_train, X_val, y_train, y_val, datagen, num_epochs, batch_size)
    


    # Create a aCTUAL MODEL 
    actual_model, act_history=create_actual_model(X_train, X_val, y_train, y_val, datagen, num_epochs, batch_size)

    
    '''classifier = KerasClassifier(model=create_model(input_shape), clip_values=(min_pixel_value, max_pixel_value))

    # Train the model
    #classifier.fit(train_images, train_labels, nb_epochs=epochs, batch_size=batch_size)
    
    history = classifier.fit(train_images, train_labels,
              batch_size=batch_size,
              nb_epochs=epochs,
              validation_data=(test_images, test_labels),
              shuffle=True)'''    

    t_consumed=(time.time() - t_start)
    print("Done...Time consumed in training:", t_consumed)
    
    return actual_model, t_consumed

def test(classifier, test_images, test_labels):
    # Evaluate the model on the test set
    test_preds = classifier.predict(test_images)
    test_labels_pred = np.argmax(test_preds, axis=1)
    test_labels_true = test_labels #np.argmax(test_labels, axis=1)

    acc = accuracy_score(test_labels_true, test_labels_pred)
    prc=precision_score(test_labels_true, test_labels_pred, average='weighted')
    rec=recall_score(test_labels_true, test_labels_pred, average='weighted')
    f1=f1_score(test_labels_true, test_labels_pred, average='weighted')

    acc=round(acc*100,2)
    prc=round(prc*100,2)
    rec=round(rec*100,2)
    f1=round(f1*100, 2)
    print('acc, prc, rec, f1:', acc, prc, rec, f1)
    
    return acc, prc, rec, f1
