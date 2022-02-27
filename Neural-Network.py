import os
from pickletools import optimize
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#mnist = tf.keras.datasets.mnist

#(x_train, y_train), (x_test, y_test) = mnist.load_data()#splitting data

# pixel has 0-255 values scale down to between 0 to 1
#x_train = tf.keras.utils.normalize(x_train, axis=1)
#x_test = tf.keras.utils.normalize(x_test, axis=1)
# creating model
#model =  tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))# creating layer and flatten the input shape naar van 28 by 28px 
# dus we hebben 28 X 28 is 784 pixels 28, 28 is een grid net als 1920 bij 1080
# nu gaan we 3 dense layers maken
#model.add(tf.keras.layers.Dense(128, activation='relu'))
#model.add(tf.keras.layers.Dense(128, activation='relu'))
#model.add(tf.keras.layers.Dense(10, activation='softmax'))


#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=3)

#model.save('handwritten.model')