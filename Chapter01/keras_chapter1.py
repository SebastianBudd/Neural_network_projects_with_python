from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(34)

model = Sequential()

# Layer 1
model.add(Dense(units=4, activation='sigmoid', input_dim=1))
# Output Layer
model.add(Dense(units=1, activation='sigmoid'))

print(model.summary())
print('')

sgd = optimizers.SGD(lr=1)
model.compile(loss='mean_squared_error', optimizer=sgd)

X = np.array([[0], [1], [2], [3]])
y = np.array([[0], [1], [0.5], [0.2]])
print(X)

model.fit(X, y, epochs=2500, verbose=False)

print(model.predict(X))
