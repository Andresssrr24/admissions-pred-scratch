from admission_model_nn import predict, scaler
import numpy as np
import pandas as pd

parameters = np.load('.../model_parameters2.npz') 

data = pd.read_csv('test_data.csv')
X_test = data.drop(columns=['Chance of Admit'])


#data = np.array(list(data.values())).reshape(1, -1)

data_norm = scaler.fit_transform(data).T

predictions = predict(data_norm, parameters)
print(predictions)

for prediction in predictions[0]: 
    decision = 'You will be admitted' if prediction > 0.5 else 'You won`t be addmited'
    print(f'Probs of admission: {prediction:.4f}')
    print(decision)
