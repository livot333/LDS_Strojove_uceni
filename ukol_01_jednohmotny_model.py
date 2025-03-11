import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 
from tensorflow import keras
from prediction_form_network_mark01 import Prediction
from neural_network_keras_seqential import SequentialNeuralNetwork
from Random_forest_regresion import RandomForestRegresion
from Benchmark import NetworkBenchmark

# Definování rozsahů parametrů
hmotnost_rozsah = [0.1, 10]
tlumeni_rozsah = [0, 5]
tuhost_rozsah = [10, 1000]
frekvence_rozsah = [0.1, 10]

# vlastnosti neuronove site
epochs = 20
validation_split = 0.2
test_size = 0.2
learning_patience = 25 #how many epochs we wait before stopping training if validation loss (MSE) does not improve.
                        #neni treba sledovat u RFR
# Konstantní amplituda budicí síly (nastavíme např. 10 N)
F_0 = 10  

# Generování náhodných dat
pocet_vzorku = 1000
hmotnost = np.random.uniform(hmotnost_rozsah[0], hmotnost_rozsah[1], pocet_vzorku)
tlumeni = np.random.uniform(tlumeni_rozsah[0], tlumeni_rozsah[1], pocet_vzorku)
tuhost = np.random.uniform(tuhost_rozsah[0], tuhost_rozsah[1], pocet_vzorku)
frekvence = np.random.uniform(frekvence_rozsah[0], frekvence_rozsah[1], pocet_vzorku)

# Výpočet amplitudy (zde je potřeba vložit váš výpočet)
# Pro zjednodušení použijeme náhodné hodnoty
omega = 2 * np.pi * frekvence  # Převod frekvence na kruhovou frekvenci
amplituda = F_0 / np.sqrt((tuhost - hmotnost * omega**2)**2 + (tlumeni * omega)**2)
# amplituda = np.random.uniform(0, 10, pocet_vzorku)

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

Model = RandomForestRegresion(x=X,y=y,estimators=epochs,test_size=test_size)  
        #RandomForestRegresion(x=X,y=y,estimators=epochs,test_size=test_size)     
        #SequentialNeuralNetwork(X,y,epochs=epochs,validation_split=validation_split,test_size=test_size,patience=learning_patience)  

scaler = Model.Scaler()
model = Model.Model()
model_loss = Model.Evaluation()
# Model.Save()
model_informations = Model.ModelInfo()

print(model_informations)

benchmark = NetworkBenchmark(model_info=model_informations)
benchmark.StoreData()


data_pro_odhad = [2.5, 1.2, 500, 5.0]

prediction = Prediction(model=model,scaler=scaler,values=data_pro_odhad,force=F_0)
odhad_amplitudy = prediction.values_predict()
print(f"Predicted Amplitude: {odhad_amplitudy}")
 
vypocet_amplitudy = prediction.calculate_values()
print(f"Analytická amplituda: {vypocet_amplitudy:.4f}")



