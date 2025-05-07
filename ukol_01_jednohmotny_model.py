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
import itertools

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
frekvence_rozsah = [0.1, 10]

# vlastnosti neuronove site
epochs = 123   # pro RFR se jedna o hodnotu number of estimators tzn pocet decision trees v modelu
validation_split = 0.2
test_size = 0.2
learning_patience = 10 #how many epochs we wait before stopping training if validation loss (MSE) does not improve.
                        #neni treba sledovat u RFR
activation_function_list = "relu"
kernel_initializer = "he_uniform"
optimizer_list = "rmsprop"
dropout_rate = 0.1 #drzet mezi 0.1 a 0.5
l1_value = 0.0001
l2_value = 0.0001


# #==================uceni===========
# neuron_options = [4,8,16,32,64]
# min_layers = 0
# max_layers = 3

# all_architectures = []

# # Procházíme počty vrstev od 2 do 6 (včetně)
# for num_layers in range(min_layers, max_layers + 1):
#     # Generujeme všechny kombinace délky num_layers z neuron_options
#     # itertools.product vrací iterátor tuplicí
#     combinations = itertools.product(neuron_options, repeat=num_layers)

#     # Přidáme každou kombinaci (převedenou na list) do hlavního seznamu
#     for combo_tuple in combinations:
#         all_architectures.append(list(combo_tuple))


hidden_layer_units_list = [16,32]#[1024,1024,1024,1024,128]
# vlastnosti learning rate scheduleru 
learn_rate_sched_patience = 12      # mensi nez learning_patience
learn_rate_sched_factor = 0.2         #idealne mezi 0.1 a 0.5


# Generování náhodných dat
pocet_vzorku = 10000
hmotnost = np.random.uniform(hmotnost_rozsah[0], hmotnost_rozsah[1], pocet_vzorku)
tlumeni = np.random.uniform(tlumeni_rozsah[0], tlumeni_rozsah[1], pocet_vzorku)
tuhost = np.random.uniform(tuhost_rozsah[0], tuhost_rozsah[1], pocet_vzorku)
frekvence = np.random.uniform(frekvence_rozsah[0], frekvence_rozsah[1], pocet_vzorku)


# Pro zjednodušení použijeme náhodné hodnoty
omega0 =  np.sqrt(tuhost/hmotnost)  
bkr = 2*np.sqrt(tuhost*hmotnost)
bp = tlumeni / bkr
omega = 2*np.pi*frekvence
ni = omega/omega0
pomerny_utlum = 1 / np.sqrt(((1 - (ni**2))**2) + (2 * bp * ni)**2)
# amplituda = np.random.uniform(0, 10, pocet_vzorku)

# data pro odhad 
#Vstupn9 data pro odhad z modelu
data_pro_odhad = [2.5 # hmotnost
                , 3 # tlumeni
                , 100 #tuhost
                , 15.0 # frekvence
                ]

omega0_pred =  np.sqrt(data_pro_odhad[2]/data_pro_odhad[0])
bkr_pred = 2*np.sqrt(data_pro_odhad[2]*data_pro_odhad[0])
bp_pred = data_pro_odhad[2] / bkr_pred
omega_pred = 2*np.pi*data_pro_odhad[3]
ni_pred = omega_pred/omega0_pred 

data_pro_odhad_komplet  =[data_pro_odhad[0],data_pro_odhad[1],data_pro_odhad[2],data_pro_odhad[3],omega0_pred,omega_pred,bkr_pred,bp_pred,ni_pred]
print(np.mean(pomerny_utlum))
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

# Rozdělení na vstupy a výstupy
X = data[["hmotnost", "tlumeni", "tuhost", "frekvence","vlastni_frekvence","vlastni_frekvence1","bkr","bp","ni"]].values
y = data["pomerny_utlum"].values


# for hidden_layer_units_list in all_architectures:
Model = SequentialNeuralNetwork(
x=X,
y=y,
epochs=epochs,
validation_split=validation_split,
test_size=test_size,
patience=learning_patience,
learn_rate_sched_factor=learn_rate_sched_factor, 
learn_rate_sched_patience=learn_rate_sched_patience,
activation_function= activation_function_list,
kernel_initializer=kernel_initializer,
optimizer=optimizer_list,
dropout_rate=dropout_rate, 
l1_value=l1_value,
l2_value=l2_value,
hidden_layer_units=hidden_layer_units_list
)  
# RandomForestRegresion(x=X,y=y,estimators=epochs,test_size=test_size)     
        
scaler = Model.Scaler()
model = Model.Model()
model_loss = Model.Evaluation()
# Model.Save()
model_informations = Model.ModelInfo()

print(model_informations)

benchmark = NetworkBenchmark(model_info=model_informations)
benchmark.StoreData()

prediction = Prediction(model=model,scaler=scaler,values=data_pro_odhad_komplet)


odhad_amplitudy = prediction.values_predict()
print(f"Predikovaný poměrný útlum: {odhad_amplitudy}")

vypocet_amplitudy = prediction.calculate_values()
print(f"Analyticky poměrný útlum: {vypocet_amplitudy:.4f}")