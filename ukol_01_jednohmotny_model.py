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
epochs = np.arange(50,121,10)               # pro RFR se jedna o hodnotu number of estimators tzn pocet decision trees v modelu
validation_split = 0.2
test_size = 0.2
learning_patience = 15 #how many epochs we wait before stopping training if validation loss (MSE) does not improve.
                        #neni treba sledovat u RFR
activation_function = ["relu", "selu", "sigmoid", "gelu", "leaky_relu", "tanh", "swish", "mish", "elu"]
kernel_initializer = "he_normal"
optimizer = ["adam","sgd","rmsprop","adagrad","adamax","nadam","adamw","lion"]
dropout_rate = 0.2      #drzet mezi 0.1 a 0.5
l1_value = 0.0001
l2_value = 0.001
hidden_layer_units = [32,64,64]

# vlastnosti learning rate scheduleru 
learn_rate_sched_patience = 10          # mensi nez learning_patience
learn_rate_sched_factor = 0.25           #idealne mezi 0.1 a 0.5


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

# Rozdělení na vstupy a výstupy
X = data[["hmotnost", "tlumeni", "tuhost", "frekvence","vlastni_frekvence","vlastni_frekvence1","bkr","bp","ni"]].values
y = data["pomerny_utlum"].values



for optimizer in optimizer:
        for activation_function in activation_function:
                for epochs in epochs:
                        Model = SequentialNeuralNetwork(
                        x=X,
                        y=y,
                        epochs=epochs,
                        validation_split=validation_split,
                        test_size=test_size,
                        patience=learning_patience,
                        learn_rate_sched_factor=learn_rate_sched_factor, 
                        learn_rate_sched_patience=learn_rate_sched_patience,
                        activation_function= activation_function,
                        kernel_initializer=kernel_initializer,
                        optimizer=optimizer,
                        dropout_rate=dropout_rate, 
                        l1_value=l1_value,
                        l2_value=l2_value,
                        hidden_layer_units=hidden_layer_units
                        )  
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



# prediction = Prediction(model=model,scaler=scaler,values=data_pro_odhad)


# odhad_amplitudy = prediction.values_predict()
# print(f"Predikovaný poměrný útlum: {odhad_amplitudy}")

# vypocet_amplitudy = prediction.calculate_values()
# print(f"Analyticky poměrný útlum: {vypocet_amplitudy:.4f}")



