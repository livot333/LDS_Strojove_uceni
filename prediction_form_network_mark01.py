import numpy as np
import joblib
from tensorflow import keras

class Prediction:
    def __init__(self, model, scaler, values, force):
        self.model = model
        self.scaler = scaler
        self.values = np.array([values])  
        self.force = force
    
    def values_predict(self):
        values_scaled = self.scaler.transform(self.values)  
        amplitude_prediction = self.model.predict(values_scaled)
        return amplitude_prediction
    
    def calculate_values(self):
        mass, damping, stiffness, frequency = self.values[0]  
        omega = 2 * np.pi * frequency
        amplitude_analytical = self.force / np.sqrt((stiffness - mass * omega**2)**2 + (damping * omega)**2)
        return amplitude_analytical  
    