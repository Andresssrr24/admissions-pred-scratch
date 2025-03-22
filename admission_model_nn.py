import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # TODO build from scratch instead
from sklearn.preprocessing import StandardScaler  # TODO build from scratch instead

# Data processing
dataset = pd.read_csv('/Users/Admin/Documents/MachineLearning/datasets/Admission_Predict_Ver1.1.csv')
# Convert the column chance of admit to 0 or 1
dataset['Chance of Admit'] = (dataset['Chance of Admit ']) #>= 0.5).astype(int)
# Load dataset
X = dataset[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']]
Y = dataset[['Chance of Admit']]
#print(X)
#print(Y)

# Data segmentation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).T  # Shape (7, 400)
X_test = scaler.transform(X_test).T  # Shape (7, 100)
Y_train = Y_train.values.reshape(1 , -1) # (1, 400)
Y_test = Y_test.values.reshape(1, -1) # (1, 100)

# /---------------------------------------/
# Activation functions (forward & back prop)
def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def sigmoid_derivative(Z):
    s = sigmoid(Z)  # Calculate sigmoid to avoid overflow due to very small or large values
    return s * (1 - s)

def relu_derivative(Z):
    return np.where(Z > 0, 1, 0)

# /---------------------------------------/
# He initialization
def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
   
    return parameters

# /---------------------------------------/
# Forward propagation
def linear_forward_prop(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache

def forward_prop_activation(A_prev, W, b, activation):
    Z, linear_cache = linear_forward_prop(A_prev, W, b)

    if activation == 'relu':
        A = relu(Z)
        activation_cache = Z

    elif activation == 'sigmoid':
       A = sigmoid(Z)
       activation_cache = Z

    else:
        raise ValueError('Unsupported activation. Try relu or sigmoid instead.')

    cache = (linear_cache, activation_cache)
    return A, cache

def forward_prop(X, parameters):
    caches= []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A

        A, cache = forward_prop_activation(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)

    AL, cache = forward_prop_activation(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)

    return AL, caches

# /---------------------------------------/
# Cost computation
def cross_entropy_with_L2(AL, Y, parameters, lambd):
    m = Y.shape[1]
    AL = np.clip(AL, 1e-10, 1 - 1e-10) # Add epsilon to avoid log(0)

    cost = -1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) 

    L2_regularization = 0
    for key in parameters.keys():
        if 'W' in key:
            L2_regularization += np.sum(np.square(parameters[key]))
    L2_regularization = (1/m) * (lambd/2) * L2_regularization

    cost_L2 = cost + L2_regularization
    cost_L2 = np.squeeze(cost_L2)

    return cost_L2



# /---------------------------------------/
# Backpropagation
def linear_backprop(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev  = np.dot(W.T, dZ)

    return dA_prev, dW, db

def backprop_activation(dA, cache, activation):
    linear_cache, activation_cache = cache
    Z = activation_cache

    if activation == 'relu':
        dZ = dA * (relu_derivative(Z))
        dA_prev, dW, db = linear_backprop(dZ, linear_cache)

    elif activation == 'sigmoid': 
        dZ = dA * (sigmoid_derivative(Z))
        dA_prev, dW, db = linear_backprop(dZ, linear_cache)

    else:
        raise ValueError('Unsupported activation. Try relu or sigmoid instead.')

    return dA_prev, dW, db
    
def backpropagation(AL, Y, caches):
    grads = {}
    
    L = len(caches)
    m = AL.shape[1]
    AL = np.clip(AL, 1e-10, 1 - 1e-10) # Add epsilon to avoid Y/0
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = backprop_activation(dAL, current_cache, 'sigmoid')
    grads['dA' + str(L-1)] = dA_prev_temp
    grads['dW' + str(L)] = dW_temp
    grads['db' + str(L)] = db_temp

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = backprop_activation(grads['dA' + str(l+1)], current_cache, 'relu')
        grads['dA' + str(l)] = dA_prev_temp
        grads['dW' + str(l+1)] = dW_temp
        grads['db' + str(l+1)] = db_temp

    return grads

# /---------------------------------------/
# Gradient descent
def grad_desc(parameters, grads, lr):
    #params = parameters.copy()
    L = len(parameters) // 2

    for l in range(L):
        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - lr * grads['dW' + str(l+1)]
        parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - lr * grads['db' + str(l+1)]

    return parameters
                                                            
# /---------------------------------------/
# Training
def train(X, Y, num_iterations, lr=0.01):
    n_0 = X_train.shape[0]
    layer_dims = [n_0, 30, 8, 5, 3, 1]  #[n_0, 10, 5, 1]  [n_0, 30, 8, 5, 3, 1]
    parameters = initialize_parameters(layer_dims)

    costs = []
    for i in range(num_iterations):
        AL, caches = forward_prop(X, parameters)
        cost = cross_entropy_with_L2(AL, Y, parameters, lambd=0.01)
        grads = backpropagation(AL, Y, caches)
        parameters = grad_desc(parameters, grads, lr)

        if i % 100 == 0:
            print(f'Cost after iteration {i}: {cost}')
            costs.append(cost) 

    plt.plot(range(0, num_iterations, 100), costs)
    plt.xlabel('Iterations (x100)')
    plt.ylabel('Cost')
    plt.title('Cost reduction over time')
    plt.show()

    return parameters

def predict(X, params):
    AL, _ = forward_prop(X, params)
    #predictions = (AL > 0.5).astype(int)

    return AL

parameters = train(X_train, Y_train, num_iterations=12000, lr=0.01)

Y_pred_train = predict(X_train, parameters)
Y_pred_test = predict(X_test, parameters)

train_mse = np.mean((Y_pred_train - Y_train) ** 2)
test_mse = np.mean((Y_pred_test - Y_test) ** 2)

print(f'\nTrain MSE: {train_mse:.4f}')
print(f'Test MSE: {test_mse:.4f}')