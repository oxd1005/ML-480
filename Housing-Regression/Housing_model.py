#you will implement linear regression with multiple variables to
#predict the prices of houses. Suppose you are selling your house and you
#want to know what a good market price would be. One way to do this is to
#first collect information on recent houses sold and make a model of housing
#prices.
#The file Housing_data.txt contains a training set of housing prices in Portland, Oregon. The first column is the size of the house (in square feet), the
#second column is the number of bedrooms, and the third column is the price of the house.
#
#  Instructions
#  ------------
# 
#  This file will give you outlines to get started on buiding a linear regression model. 
#

## ================ Part 1: Feature Normalization ================

## Load Housing_data.txt
import os
import pandas as pd

file_data = open(os.path.join(os.path.dirname(__file__), 'Housing_data.txt'), 'r')
data = pd.read_csv(file_data, names=['Size', 'Bedrooms', 'Price'])


####################################################################

# Print out some data points
print(data.head())


####################################################################


# Scale features and set them to zero mean
# ====================== YOUR CODE HERE ======================
#               First, for each feature dimension, compute the mean
#               of the feature and subtract it from the dataset.  
#               Next, compute the standard deviation of each feature and divide
#               Next, You need to perform the normalization separately for each feature. 

for col_index in range(0, len(data.columns)-1):
    col_data = data.iloc[:,col_index]
    col_data = (col_data - col_data.mean())/col_data.std()
    data.iloc[:,col_index] = col_data


####################################################################

# Print out some data points
print(data.head())


####################################################################



## =================== Part 2: Cost and Gradient descent ===================

import numpy as np

# Some gradient descent settings
iterations = 1500;
alpha = 0.01;
m = len(data)

# Init Theta and Run Gradient Descent 
def compute_cost(X, y, theta):
    """
    Compute cost for linear regression.
    """
    predictions = X.dot(theta)
    errors = predictions - y
    J = (1 / (2 * m)) * np.sum(errors ** 2)
    
    return J

X = np.column_stack((np.ones(data.shape[0]), data[['Size', 'Bedrooms']].values))
y = data['Price'].values
theta = np.zeros(X.shape[1])
print(X.shape, y.shape, theta.shape)

# Compute and display the initial cost with theta initialized to zeros
initial_cost = compute_cost(X, y, theta)
print(initial_cost)

# Perform a single gradient step on the parameter thetas. 
for t in range(len(theta)):
    partial_derivative = (1 / m) * np.sum((np.dot(X, theta) - y) * X[:,t])
    theta[t] = theta[t] - alpha * partial_derivative

print(f'Cost after one iteration: {compute_cost(X, y, theta)}')

#   After updating theta, compute the cost of using the current values of theta to fit the data points 
#   You should set J_history(iter) to the cost of using the current values of theta. 
#   Save the cost J in every iteration    
J_history = np.zeros(iterations)

for iter in range(iterations):
    predictions = X.dot(theta)
    errors = predictions - y
    for t in range(len(theta)):
        partial_derivative = (1 / m) * np.sum(errors * X[:,t])
        theta[t] = theta[t] - alpha * partial_derivative
    
    J_history[iter] = compute_cost(X, y, theta)

print(f'Cost after {iterations} iterations: {J_history[-1]}')
    
####################################################################

# Plot the convergence graph
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(iterations), J_history, color='blue')
plt.title('Convergence of Cost Function')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.grid(True)

####################################################################

# Display gradient descent's result
plt.show()

####################################################################

# Estimate the price of a 1650 sq-ft, 3 br house
# Feature normalization for the new data point
size_normalized = (1650 - data['Size'].mean()) / data['Size'].std()
bedrooms_normalized = (3 - data['Bedrooms'].mean()) / data['Bedrooms'].std()

# Predicting the price using the trained model
price_prediction = np.dot([1, size_normalized, bedrooms_normalized], theta)
print(f'Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${price_prediction:.2f}')

####################################################################

