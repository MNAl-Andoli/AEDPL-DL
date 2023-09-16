import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from art.estimators.classification import KerasClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

import time


# Define the CNN architecture
def create_model(input_shape):
    '''model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    #model.add(layers.Dense(64, activation='relu',name='dense_1'))
    model.add(layers.Dense(32, activation='relu',name='dense_2'))
    model.add(layers.Dense(10, activation='softmax'))'''
    
    INPUT_SHAPE = (32, 32, 3)
    KERNEL_SIZE = (3, 3)
    model = Sequential()

    # Convolutional Layer
    model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
    model.add(BatchNormalization())
    # Pooling layer
    model.add(MaxPool2D(pool_size=(2, 2)))
    # Dropout layers
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    #model.add(Dense(64, activation='relu',name='dense_1'))
    model.add(Dense(32, activation='relu',name='dense_features'))

    model.add(Dropout(0.25))
    model.add(Dense(11, activation='softmax'))



    # CNN model, with 6 convolutional layers, 3 pooling layers, and 3 dense layers. Softmax output layer.
    '''model = models.Sequential()

    model.add(layers.Conv2D(64, (5,5), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))'''
    
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


	


    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

def train(train_images, train_labels, epochs, batch_size, min_pixel_value, max_pixel_value, test_images, test_labels):
    t_start=time.time()

    # Create a KerasClassifier instance
    input_shape=train_images[0].shape
    classifier = KerasClassifier(model=create_model(input_shape), clip_values=(min_pixel_value, max_pixel_value))

    # Train the model
    #classifier.fit(train_images, train_labels, nb_epochs=epochs, batch_size=batch_size)
    
    history = classifier.fit(train_images, train_labels,
              batch_size=batch_size,
              nb_epochs=epochs,
              validation_data=(test_images, test_labels),
              shuffle=True)    

    t_consumed=(time.time() - t_start)
    print("Done...Time consumed in training:", t_consumed)
    
    return classifier, t_consumed

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
