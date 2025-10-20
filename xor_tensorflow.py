from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import numpy as np
import random
import tensorflow as tf

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Inputs and outputs
A = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])
b = np.array([0, 1, 1, 0])

# Creating the model
# Use two extra neurons to converge faster, but set learning_rate lower
x = Input(shape=(2,))
h = Dense(4, activation="sigmoid")(x)
y = Dense(1, activation="sigmoid")(h)
model = Model(x, y)
optimizer = Adam(learning_rate=0.1)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Fitting the model
model.fit(A, b, epochs=100)

# Getting model weights
for layer_idx, layer in enumerate(model.layers):
    weights = layer.get_weights()
    print("\nLayer {}: {}".format(layer_idx, layer.name))
    if weights:
        kernel, bias = weights
        print("Weights (kernel):\n", kernel)
        print("Bias:\n", bias)
    else:
        print("No weights in this layer.")

# Checking model predictions
b_pred = np.round(model.predict(A))
print(classification_report(b, b_pred))
