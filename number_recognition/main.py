#!usr/bin/python3

"""The MIT License (MIT)

Copyright (c) 2021 mauro-balades <mauro.balades@tutanota.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

class NumberRecognizer:

    model = 'handwritten.model'

    def __init__(self, model: str = model):

        self._mnist = tf.keras.datasets.mnist
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self._mnist.load_data()

        self.x_train = tf.keras.utils.normalize(self.x_train, axis=1)
        self.x_test = tf.keras.utils.normalize(self.x_test, axis=1)

        self.model = model

    def init(self, epochs: int = 10):

        _model = tf.keras.models.Sequential()
        _model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
        _model.add(tf.keras.layers.Dense(128, activation='relu'))
        _model.add(tf.keras.layers.Dense(128, activation='relu'))
        _model.add(tf.keras.layers.Dense(10, activation='softmax'))

        _model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        _model.fit(self.x_train, self.y_train, epochs=epochs)

        _model.save(self.model)

    def load(self):

        self._model = tf.keras.models.load_model(self.model)
        loss, accuracy = self._model.evaluate(self.x_test, self.y_test)

        return (loss, accuracy)

    def recognize(self, path: str):

        try:
            img = cv2.imread(path)[:,:,0]
            img = np.invert(np.array([img]))
            prediction = self._model.predict(img)
            return np.argmax(prediction)
        except Exception as e:
            print(f"Error: {e}")
            
