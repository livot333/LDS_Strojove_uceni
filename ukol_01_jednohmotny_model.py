import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import joblib 

# Definování rozsahů parametrů
hmotnost_rozsah = [0.1, 10]
tlumeni_rozsah = [0, 5]
tuhost_rozsah = [10, 1000]
frekvence_rozsah = [0.1, 10]

# Konstantní amplituda budicí síly (nastavíme např. 10 N)
F_0 = 10  

# Generování náhodných dat
pocet_vzorku = 10000
hmotnost = np.random.uniform(hmotnost_rozsah[0], hmotnost_rozsah[1], pocet_vzorku)
tlumeni = np.random.uniform(tlumeni_rozsah[0], tlumeni_rozsah[1], pocet_vzorku)
tuhost = np.random.uniform(tuhost_rozsah[0], tuhost_rozsah[1], pocet_vzorku)
frekvence = np.random.uniform(frekvence_rozsah[0], frekvence_rozsah[1], pocet_vzorku)

# Výpočet amplitudy (zde je potřeba vložit váš výpočet)
# Pro zjednodušení použijeme náhodné hodnoty
# omega = 2 * np.pi * frekvence  # Převod frekvence na kruhovou frekvenci
# amplituda = F_0 / np.sqrt((tuhost - hmotnost * omega**2)**2 + (tlumeni * omega)**2)
amplituda = np.random.uniform(0, 10, pocet_vzorku)

# Vytvoření Data
data = pd.DataFrame({
    "hmotnost": hmotnost,
    "tlumeni": tlumeni,
    "tuhost": tuhost,
    "frekvence": frekvence,
    "amplituda": amplituda
        })

# Rozdělení na vstupy a výstupy
X = data[["hmotnost", "tlumeni", "tuhost", "frekvence"]].values
y = data["amplituda"].values

# Rozdělení na trénovací a testovací množinu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizace dat
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)

# Návrh modelu
model = keras.Sequential([
    keras.layers.Dense(512, activation="sigmoid", input_shape=(X_train.shape[1],)),
    keras.layers.Dense(512, activation="sigmoid"),
    keras.layers.Dense(1)])

# Kompilace modelu
model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
# Trénování modelu
model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Hodnocení modelu
loss = model.evaluate(X_test, y_test)
print(f"Testovací MSE: {loss}")
 

# Save the model and scaler
model.save("jednohmotny_model_mark01.h5")
joblib.dump(scaler, "scaler_4_mark01.pkl")

print("Model and scaler saved successfully.")


