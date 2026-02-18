import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Sample training data
X = np.array([
    [100, 5],
    [200, 7],
    [300, 9],
    [400, 11],
    [500, 13]
])

y = np.array([120, 250, 380, 520, 680])

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved successfully!")