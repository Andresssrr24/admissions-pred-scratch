import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Data processing
dataset = pd.read_csv('.../Admission_Predict_Ver1.1.csv')
dataset['Chance of Admit'] = (dataset['Chance of Admit '])

# Split dataset (X, Y)
X = dataset[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']]
Y = dataset[['Chance of Admit']]

# Randomize data order
dataset = dataset.sample(frac=1)

# Split dataset (train, test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
'''train_sets_ratio = 0.8
test_sets_ratio = 0.2'''

'''X_train = X[:int(len(X) * train_sets_ratio)]
Y_train = Y[:int(len(Y) * train_sets_ratio)]

X_test =  X[:int(len(X) * test_sets_ratio)]
Y_test =  Y[:int(len(Y) * test_sets_ratio)] '''

# /---------------------------------------/
# Input normalization

scaler = ColumnTransformer(
    transformers=[
        ('numeric_features', StandardScaler(), ['GRE Score', 'TOEFL Score', 'CGPA']),
        ('ordinal_fatures', MinMaxScaler(), ['University Rating', 'SOP', 'LOR ']),
        ('binary_features', 'passthrough', ['Research'])
    ])

X_train = scaler.fit_transform(X_train).T  # Shape (7, 400)
X_test = scaler.fit_transform(X_test).T  # Shape (7, 100)
Y_train = Y_train.values.reshape(1 , -1) # (1, 400)
Y_test = Y_test.values.reshape(1, -1) # (1, 100)'''

'''def input_normalization(X):
    data_min = np.min(X)
    data_max = np.max(X)
    X_norm = (X - data_min) / (data_max - data_min + 1e-8)

    return X_norm'''

'''X_train = input_normalization(X_train).T  # Shape (7, 400)
X_test = input_normalization(X_test).T  # Shape (7, 100)
Y_train = Y_train.values.reshape(1 , -1) # (1, 400)
Y_test = Y_test.values.reshape(1, -1) # (1, 100)'''

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
def mean_squared_error_with_L2(AL, Y, parameters, lambd):
    m = Y.shape[1]
    AL = np.clip(AL, 1e-10, 1 - 1e-10) # Add epsilon to avoid log(0)

    cost = (1 / (2 * m)) * np.sum((AL - Y) ** 2)

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


def adam_optimization(parameters, grads, t, lr, beta1, beta2, epsilon):
    L = len(parameters) // 2
    v = {}
    s = {}
    v_corrected = {}
    s_corrected = {}

    for l in range(1, L + 1):
        # Adam moments initialization
        v['dW' + str(l)] = np.zeros((parameters["W" + str(l)].shape))
        v['db' + str(l)] = np.zeros((parameters['b' + str(l)].shape))
        s['dW' + str(l)] = np.zeros((parameters['W' + str(l)].shape))
        s['db' + str(l)] = np.zeros((parameters['b' + str(l)].shape))

    t += 1

    for l in range(1, L + 1):
        # Moment #1
        v['dW' + str(l)] = beta1 * v['dW' + str(l)] + (1 - beta1) * grads['dW' + str(l)]
        v['db' + str(l)] = beta1 * v['db' + str(l)] + (1 - beta1) * grads['db' + str(l)]

        v_corrected['dW' + str(l)] = v['dW' + str(l)] / (1 - (beta1 ** t))
        v_corrected['db' + str(l)] = v['db' + str(l)] / (1 - (beta1 ** t))

        # Moment #2
        s['dW' + str(l)] = beta2 * s['dW' + str(l)] + (1 - beta2) * np.square(grads['dW' + str(l)])
        s['db' + str(l)] = beta2 * s['db' + str(l)] + (1 - beta2) * np.square(grads['db' + str(l)])

        s_corrected['dW' + str(l)] = s['dW' + str(l)] / (1 - (beta2 ** t))
        s_corrected['db' + str(l)] = s['db' + str(l)] / (1 - (beta2 ** t))

        # Parameter update
        parameters['W' + str(l)] = parameters['W' + str(l)] - lr * (v_corrected['dW' + str(l)] / (np.sqrt(s_corrected['dW' + str(l)]) + epsilon))
        parameters['b' + str(l)] = parameters['b' + str(l)] - lr * (v_corrected['db' + str(l)] / (np.sqrt(s_corrected['db' + str(l)]) + epsilon))

    return parameters, s, v, s_corrected, v_corrected

# /---------------------------------------/
# Training
def train(X, Y, num_iterations, lr=0.0002): # 1(0.0002)
    n_0 = X_train.shape[0]
    layer_dims = [n_0, 7, 5, 3, 1]  # 1[n_0, 7, 5, 3, 1]  2[n_0, 7, 7, 3, 1]
    parameters = initialize_parameters(layer_dims)

    costs = []
    t = 0
    for i in range(num_iterations):
        AL, caches = forward_prop(X, parameters)
        cost = mean_squared_error_with_L2(AL, Y, parameters, lambd=0.00003) # 1(0.00003)
        grads = backpropagation(AL, Y, caches)
        parameters, _, _, _, _ = adam_optimization(parameters, grads, t, lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

        if i % 100 == 0:
            print(f'Cost after iteration {i}: {cost}')
            costs.append(cost) 

    plt.plot(range(0, num_iterations, 100), costs)
    plt.xlabel('Iterations (x100)')
    plt.ylabel('Cost')
    plt.title('Cost reduction over time')
    plt.show()

    return parameters

# /---------------------------------------/
# Post training process
# Predictions
def predict(X, params):
    AL, _ = forward_prop(X, params)

    return AL

# /---------------------------------------/
if __name__ == '__main__':
    parameters = train(X_train, Y_train, num_iterations=15000, lr=0.0002) # 1(15000, 0.0002)

    Y_pred_train = predict(X_train, parameters)
    Y_pred_test = predict(X_test, parameters)

    train_mse = np.mean((Y_pred_train - Y_train) ** 2)
    test_mse = np.mean((Y_pred_test - Y_test) ** 2)

    print(f'\nTrain MSE: {train_mse:.4f}')
    print(f'Test MSE: {test_mse:.4f}')

    # /---------------------------------------/ 
    # Save model
    np.savez('.../model_parameters2.npz', **parameters)