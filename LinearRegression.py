from sklearn.linear_model import LinearRegression
import numpy as np

# Generating sample data for training
x_train = np.array([[1], [2], [3], [4]])
y_train = np.array([2, 4, 6, 8])

# Creating a Linear Regression model object
reg = LinearRegression().fit(x_train, y_train)

# Predicting a new value using the trained model
x_test = np.array([[5]])
y_pred = reg.predict(x_test)

print("Prediction for x = 5: ", y_pred)
