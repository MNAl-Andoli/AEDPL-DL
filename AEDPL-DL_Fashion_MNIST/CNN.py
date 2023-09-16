import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from art.estimators.classification import KerasClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

lr=0.001

# Define the CNN architecture
def create_model(input_shape):
    # Build the neural network model (CNN)
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu',name='dense_2'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model


    
def train(train_images, train_labels, epochs, batch_size, min_pixel_value, max_pixel_value):
    t_start=time.time()

    # Create a KerasClassifier instance
    input_shape=train_images[0].shape
    classifier = KerasClassifier(model=create_model(input_shape), clip_values=(min_pixel_value, max_pixel_value))

    # Train the model
    classifier.fit(train_images, train_labels, nb_epochs=epochs, batch_size=batch_size)

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
