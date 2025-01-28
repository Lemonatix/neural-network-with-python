import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(64, input_shape=(784,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

import numpy as np
x_train = np.random.random((60000, 784))
y_train = np.random.randint(10, size=(60000,))

model.fit(x_train, y_train, epochs=5, batch_size=32)