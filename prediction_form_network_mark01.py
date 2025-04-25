import numpy as np
import joblib
from tensorflow import keras

class Prediction:
    def __init__(self, model, scaler, values):
        self.model = model
        self.scaler = scaler
        self.values = np.array([values])  
    
    def values_predict(self):
        values_scaled = self.scaler.transform(self.values)  
        amplitude_prediction = self.model.predict(values_scaled)
        return amplitude_prediction
    
    def calculate_values(self):
        hmotnost, tlumeni, tuhost, frekvence = self.values[0]
        
        # Převod frekvence na kruhovou frekvenci
        omega0 = np.sqrt(tuhost / hmotnost)
        bkr = 2 * np.sqrt(tuhost * hmotnost)
        bp = tlumeni / bkr
        omega = 2 * np.pi * frekvence
        ni = omega / omega0
        
        # Výpočet amplitudy
        pomerny_utlum = 1 / np.sqrt(((1 - (ni**2))**2) + (2 * bp * ni)**2)
        
        return pomerny_utlum
    