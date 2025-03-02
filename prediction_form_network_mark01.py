import numpy as np
import joblib
from tensorflow import keras

# Load the trained model and scaler
model = keras.models.load_model("jednohmotny_model_mark01.h5")
scaler = joblib.load("scaler_4_mark01.pkl")

# Define new input data (example: mass=2.5, damping=1.2, stiffness=500, frequency=5.0)
my_values = np.array([[2.5, 1.2, 500, 5.0]])
F_0 = 10
# Normalize the input using the saved scaler
my_values_scaled = scaler.transform(my_values)

# Predict amplitude
predicted_amplitude = model.predict(my_values_scaled)

print(f"Predicted Amplitude: {predicted_amplitude[0][0]:.4f}")

# calculated amplitude
omega = 2* np.pi* my_values[0,3]
amplituda_anal = amplituda_anal = F_0 / np.sqrt((my_values[0, 2] - my_values[0, 0] * omega**2)**2 + (my_values[0, 1] * omega)**2)

print(f"amplituda analyticky: {amplituda_anal}")