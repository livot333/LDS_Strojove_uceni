import numpy as np
import matplotlib
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from tensorflow import keras
import joblib 


class SequentialNeuralNetwork():
    def __init__(self,x,y,epochs,validation_split, test_size):
        self.X = x
        self.y = y
        self.epochs = epochs
        self.validation_split = validation_split
        self.test_size = test_size
        self.model = None 

    def Scaler(self):
        # Rozdělení na trénovací a testovací množinu
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=42)

        # Normalizace dat
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train) 
        self.X_test = scaler.transform(self.X_test)
        self.scaler = scaler
        return scaler
    
    def Model(self):
        model = keras.Sequential([
                    keras.layers.Dense(512, activation="sigmoid", input_shape=(self.X_train.shape[1],)),
                    keras.layers.Dense(512, activation="sigmoid"),
                    keras.layers.Dense(1)])

        # Kompilace modelu
        model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
        # Trénování modelu
        model.fit(self.X_train, self.y_train, epochs=self.epochs, validation_split=self.validation_split)

        self.model = model
        
        print("Learning is finished.")

        return model
    

    
    def Evaluation(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call 'Model' method first.")

        # Evaluate the model on test data
        loss = self.model.evaluate(self.X_test, self.y_test)
        print(f"Test MSE: {loss}")
        return loss
    
    def Save(self):
        self.model.save("jednohmotny_model_mark01.keras")
        joblib.dump(self.scaler, "scaler_4_mark01.pkl")
        print("Model and scaler saved successfully.")

    
    def ModelInfo(self):
        # Ensure the model is available before fetching info
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call 'Model' method first.")

        # Get the model details
        model_info = {}


        # Model name (class name)
        model_info['model_name'] = self.model.__class__.__name__
        model_info['scaler_name'] = self.scaler.__class__.__name__

        # Number of layers and neurons
        model_info['layers'] = []
        for layer in self.model.layers:
            layer_info = {}
            layer_info['layer_name'] = layer.__class__.__name__
            if isinstance(layer, keras.layers.Dense):
                layer_info['num_neurons'] = layer.units  # Get the number of neurons directly
                layer_info['activation_function'] = layer.get_config().get('activation', None)
            else:
                layer_info['num_neurons'] = None
                layer_info['activation_function'] = None
            model_info['layers'].append(layer_info)

        # Optimizer name
        model_info['optimizer'] = self.model.optimizer.__class__.__name__

        return model_info
        

        