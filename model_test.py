from admission_model_nn import predict, scaler
import numpy as np
import pandas as pd

parameters = np.load('model_parameters.npz') 

data = pd.read_csv('test_data.csv')
X_test = data.drop(columns=['Chance of Admit'])

data_norm = scaler.fit_transform(data).T

predictions = predict(data_norm, parameters)
print(predictions)

for prediction in predictions[0]: 
    decision = 'You will be admitted' if prediction > 0.5 else 'You won`t be addmited'
    print(f'Probs of admission: {prediction:.4f}')
    print(decision)

''' 
   model_parameters.npz
   [[0.74895173 0.60281694 0.8705349  0.45915838 0.75512821 0.93211954]]
    
   model_parameters2.npz
   [[0.7443605  0.59343562 0.87542345 0.44932101 0.71337261 0.92803901]]

   model_parameters_3.npz
   [[0.73613267 0.55606715 0.8756744  0.53545731 0.72357552 0.93333454]]

   model_parameters_4.npz
   [[0.74307945 0.57534208 0.86393991 0.52168126 0.72257027 0.93146847]]
   
   target
   [[0.75 0.56 0.92 0.35 0.68 1]]
    
    '''