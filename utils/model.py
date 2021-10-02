import tensorflow as tf
import logging


def create_model(h1,h2,h3):
    LAYERS = [
        tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
        tf.keras.layers.Dense(h1, activation="relu", name="hiddenLayer1"),
        tf.keras.layers.Dense(h2, activation="relu", name="hiddenLayer2"),
        tf.keras.layers.Dense(h3, activation="softmax", name="outputLayer")
        ]
    model_clf = tf.keras.models.Sequential(LAYERS)
    model_clf.layers
    model_clf.summary()
    return model_clf



    





