import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

class RandomForestRegresion():
    def __init__(self, x, y, estimators, test_size):
        self.X = x
        self.y = y
        self.estimators = estimators
        self.test_size = test_size
        self.model = None
        self.history = None  # To store training history
        self.training_time = None  # To store total training time

    def Scaler(self):
        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=42)

        # Normalize the data
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train) 
        self.X_test = scaler.transform(self.X_test)
        self.scaler = scaler
        return scaler
    
    def Model(self): 
        # Start tracking time before fitting the model
        start_time = time.time()

        model = RandomForestRegressor(n_estimators=self.estimators, random_state=42)
        model.fit(self.X_train, self.y_train)

        # End tracking time after fitting the model
        end_time = time.time()
        self.training_time = end_time - start_time  # Total training time

        self.model = model
        print(f"Model training completed in {self.training_time:.2f} seconds.")
        return model
    
    def Evaluation(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call 'Model' method first.")

        # Predict on the test set
        y_pred = self.model.predict(self.X_test)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(self.y_test, y_pred)
        self.loss = mse
        # Print the evaluation result
        print(f"Test MSE: {mse:.4f}")
        return mse  # Return the MSE value
    
    def Save(self):
        joblib.dump(self.model, "random_forest_model.pkl")
        joblib.dump(self.scaler, "scaler_rf.pkl")
        print("Model and scaler saved successfully.")

    def ModelInfo(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call 'Model' method first.")

        # Get the model details
        model_info = {}

        # Model name (class name)
        model_info['model_name'] = self.model.__class__.__name__
        model_info['scaler_name'] = self.scaler.__class__.__name__

        # Number of estimators (trees)
        model_info['n_estimators'] = self.model.n_estimators

        # Hyperparameters used for the RandomForestRegressor
        model_info['max_depth'] = self.model.max_depth
            # Hyperparameters used for the RandomForestRegressor
        model_info['min_samples_split'] = self.model.min_samples_split
        model_info['min_samples_leaf'] = self.model.min_samples_leaf


        # Add training time to the model info
        model_info['training_time'] = self.training_time  # Include training time in the model info
        model_info['mse_loss'] = self.loss

        return model_info
