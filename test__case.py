import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 
from tensorflow import keras
from prediction_form_network_mark01 import Prediction
from neural_network_keras_seqential import SequentialNeuralNetwork
from Random_forest_regresion import RandomForestRegresion
from Benchmark import NetworkBenchmark
import os
import random
import tensorflow as tf 
import matplotlib.pyplot as plt

my_seed = 42
os.environ['PYTHONHASHSEED'] = str(my_seed) 
random.seed(my_seed) 
np.random.seed(my_seed) 
tf.random.set_seed(my_seed)
tf.keras.utils.set_random_seed(my_seed)


# Definování rozsahů parametrů
hmotnost_rozsah = [0.1, 10]
tlumeni_rozsah = [0, 5]
tuhost_rozsah = [10, 1000]
frekvence_rozsah = [7, 17]

# vlastnosti neuronove site
epochs = 50              # pro RFR se jedna o hodnotu number of estimators tzn pocet decision trees v modelu
validation_split = 0.2
test_size = 0.2
learning_patience = 15 #how many epochs we wait before stopping training if validation loss (MSE) does not improve.
                        #neni treba sledovat u RFR
activation_function = "relu"
kernel_initializer = "he_normal"
optimizer = "adam"
dropout_rate = 0.2      #drzet mezi 0.1 a 0.5
l1_value = 0.0001
l2_value = 0.001
hidden_layer_units = [32,128,128]

# vlastnosti learning rate scheduleru 
learn_rate_sched_patience = 10          # mensi nez learning_patience
learn_rate_sched_factor = 0.1          #idealne mezi 0.1 a 0.5


#Vstupn9 data pro odhad z modelu
data_pro_odhad = [2.5, 1.2, 500, 5.0]

# Generování náhodných dat
pocet_vzorku = 10000
hmotnost = np.random.uniform(hmotnost_rozsah[0], hmotnost_rozsah[1], pocet_vzorku)
tlumeni = np.random.uniform(tlumeni_rozsah[0], tlumeni_rozsah[1], pocet_vzorku)
tuhost = np.random.uniform(tuhost_rozsah[0], tuhost_rozsah[1], pocet_vzorku)
frekvence = np.random.uniform(frekvence_rozsah[0], frekvence_rozsah[1], pocet_vzorku)

# Výpočet amplitudy (zde je potřeba vložit váš výpočet)
# Pro zjednodušení použijeme náhodné hodnoty
omega0 =  np.sqrt(tuhost/hmotnost)  
bkr = 2*np.sqrt(tuhost*hmotnost)
bp = tlumeni / bkr
omega = 2*np.pi*frekvence
ni = omega/omega0
pomerny_utlum = 1 / np.sqrt(((1 - (ni**2))**2) + (2 * bp * ni)**2)
# amplituda = np.random.uniform(0, 10, pocet_vzorku)

# Vytvoření Data
data = pd.DataFrame({
    "hmotnost": hmotnost,
    "tlumeni": tlumeni,
    "tuhost": tuhost,
    "frekvence": frekvence,
    "pomerny_utlum": pomerny_utlum,
    "vlastni_frekvence": omega0,
    "vlastni_frekvence1":omega,
    "bkr":bkr,
    "bp":bp,
    "ni":ni
        })

# --- KÓD PRO ZOBRAZENÍ 3D GRAFŮ ---

print("\nVytváření 3D grafů pro vizualizaci závislostí...")

# Graf 1: Závislost na Frekvenci a Tlumení (původní vstupy)
fig1 = plt.figure(figsize=(10, 7))
ax1 = fig1.add_subplot(111, projection='3d') # Přidání 3D subplotu

# Vykreslení bodů. Barva bodů může reprezentovat hodnotu poměrného útlumu
scatter1 = ax1.scatter(data["frekvence"], data["tlumeni"], data["pomerny_utlum"], c=data["pomerny_utlum"], cmap='viridis', marker='.')

# Nastavení popisků os a názvu grafu
ax1.set_xlabel("Frekvence")
ax1.set_ylabel("Tlumeni")
ax1.set_zlabel("Pomerny utlum")
ax1.set_title("Zavislost Pomerny utlum na Frekvenci a Tlumeni")

# Přidání barevné legendy (colorbar)
fig1.colorbar(scatter1, label="Pomerny utlum")


# Graf 2: Závislost na Poměru frekvencí (ni) a Poměrném útlumu (bp) (vypočítané rysy)
fig2 = plt.figure(figsize=(10, 7))
ax2 = fig2.add_subplot(111, projection='3d')

scatter2 = ax2.scatter(data["ni"], data["bp"], data["pomerny_utlum"], c=data["pomerny_utlum"], cmap='viridis', marker='.')

ax2.set_xlabel("Pomer frekvenci (ni)")
ax2.set_ylabel("Pomerny utlum systemu (bp)") # Přesnější popisek pro bp
ax2.set_zlabel("Pomerny utlum (cil)")
ax2.set_title("Zavislost Pomerny utlum na Pomeru frekvenci (ni) a Pomernem utlumu systemu (bp)")

fig2.colorbar(scatter2, label="Pomerny utlum")


# Graf 3: Závislost na Tuhosti a Hmotnosti (původní vstupy)
fig3 = plt.figure(figsize=(10, 7))
ax3 = fig3.add_subplot(111, projection='3d')

scatter3 = ax3.scatter(data["tuhost"], data["hmotnost"], data["pomerny_utlum"], c=data["pomerny_utlum"], cmap='viridis', marker='.')

ax3.set_xlabel("Tuhost")
ax3.set_ylabel("Hmotnost")
ax3.set_zlabel("Pomerny utlum")
ax3.set_title("Zavislost Pomerny utlum na Tuhosti a Hmotnosti")

fig3.colorbar(scatter3, label="Pomerny utlum")


# Zobraz všechny vytvořené grafy
plt.show()
