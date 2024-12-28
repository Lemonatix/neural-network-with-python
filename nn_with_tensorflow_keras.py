import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# generate synthetic data: y=3x+2
np.random.seed(42)
x_data = np.linspace(-10, 10, 100)
y_data = 3 * x_data + 2 + np.random.normal(0, 2, size=x_data.shape)

# split into training and testing
train_ratio = 0.8
train_size = int(len(x_data) * train_ratio)

x_train = x_data[:train_size]
y_train = y_data[:train_size]
x_test = x_data[train_size:]
y_test = y_data[train_size:]

# build sequential model
model = keras.Sequential([
    keras.Input(shape=(1,)), # 1D input
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1)  # single output
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=8, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Make predictions
predictions = model.predict(x_test)
for x_val, pred_val, true_val in zip(x_test[:5], predictions[:5], y_test[:5]):
    print(f"x={x_val:.2f} | Predicted={pred_val[0]:.2f} | Actual={true_val:.2f}")

'''
if you don't use a graphics card with cuda cores to run the file, you will get an error message:
    to rid the error, do pip install tensorflow-cpu, but this is optional
    the file itself runs normal without the install
'''