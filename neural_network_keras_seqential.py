import time
import os
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")  # Use TkAgg for real-time updates
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import regularizers
import tensorflow as tf

## ===== normalizace======
my_seed = 42
os.environ['PYTHONHASHSEED'] = str(my_seed) 
random.seed(my_seed) 
np.random.seed(my_seed) 
tf.random.set_seed(my_seed)
tf.keras.utils.set_random_seed(my_seed)


# Custom callback to track time per epoch
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.time()
        self.epoch_times.append(epoch_end_time - self.epoch_start_time)

    def get_epoch_times(self):
        return self.epoch_times



class LivePlotCallback(keras.callbacks.Callback):
    def __init__(self,patience = 5):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.patience  = patience
        self.best_val_loss = float('inf')
        self.wait = 0  # Counter for patience

        # Set up live plot
        # plt.ion()
        # self.fig, self.ax = plt.subplots()
        # self.line1, = self.ax.plot([], [], label="Train MSE", color="blue")
        # self.line2, = self.ax.plot([], [], label="Validation MSE", color="red")
        # self.ax.set_xlabel("Epochs")
        # self.ax.set_ylabel("MSE Loss")
        # self.ax.legend()
        # self.ax.set_title("Training Progress")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        if train_loss is not None and val_loss is not None:
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Update plot
            # self.line1.set_xdata(range(1, len(self.train_losses) + 1))
            # self.line1.set_ydata(self.train_losses)
            # self.line2.set_xdata(range(1, len(self.val_losses) + 1))
            # self.line2.set_ydata(self.val_losses)
            # self.ax.relim()
            # self.ax.autoscale_view()
            # # plt.draw()
            # # plt.pause(0.1)  # Pause for real-time updating

            # Check for early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.wait = 0  # Reset patience counter
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch + 1} (Validation MSE stopped improving)")
                    self.model.stop_training = True

    # def on_train_end(self, logs=None):
    #     plt.ioff()
    #     plt.close(self.fig)



# neural network
class SequentialNeuralNetwork():
    def __init__(self, x, y, epochs, validation_split, test_size, patience,learn_rate_sched_factor,learn_rate_sched_patience,activation_function,kernel_initializer,optimizer,dropout_rate,l1_value,l2_value,hidden_layer_units):
        self.X = x
        self.y = y
        self.epochs = epochs
        self.validation_split = validation_split
        self.test_size = test_size
        self.patience = patience
        self.model = None
        self.history = None  # To store training history
        self.Learning_rate_scheduler = None
        self.time_history = TimeHistory()  # Instantiate the custom callback
        self.learn_rate_sched_factor = learn_rate_sched_factor
        self.learn_rate_sched_patience = learn_rate_sched_patience
        self.activation_function = activation_function 
        self.kernel_initializer = kernel_initializer
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.l1_value = l1_value
        self.l2_value = l2_value
        self.hidden_layer_units = hidden_layer_units


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
        # Vytvoříme instanci regularizátoru (stejná logika jako dříve)
        regularizer = None
        if self.l1_value > 0 or self.l2_value > 0:
             regularizer = regularizers.L1L2(l1=self.l1_value, l2=self.l2_value)

        sample_weights = np.ones(len(self.y_train)) # Výchozí váha 1 pro všechny
        resonance_threshold = 1# Stejný práh
        resonance_threshold_high = 1.1# Stejný práh
        low_weight = 0.0001 # Váha pro body nad prahem
        lowest_weights = 0.0000001

        sample_weights[self.y_train >= resonance_threshold] = low_weight
        sample_weights[self.y_train >= resonance_threshold_high] = lowest_weights

        model = keras.Sequential()
        for i, units in enumerate(self.hidden_layer_units):
            # Přidáme Dense vrstvu
            if i == 0: # První skrytá vrstva potřebuje input_shape
                model.add(keras.layers.Dense(units=units, # Používáme počet neuronů z aktuální pozice v seznamu
                                activation=self.activation_function,
                                input_shape=(self.X_train.shape[1],), # Input shape jen u první vrstvy
                                kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=regularizer))
            else: # Ostatní skryté vrstvy
                model.add(keras.layers.Dense(units=units, # Používáme počet neuronů z aktuální pozice v seznamu
                                activation=self.activation_function,
                                kernel_initializer=self.kernel_initializer,
                                kernel_regularizer=regularizer))

            model.add(keras.layers.BatchNormalization())

            if self.dropout_rate > 0:
                 model.add(keras.layers.Dropout(rate=self.dropout_rate))

        model.add(keras.layers.Dense(units=1,activation="softplus")) # Volitelná regularizace i na výstupu


        model.compile(optimizer=self.optimizer, loss=keras.losses.MeanSquaredError())           #,metrics=[keras.metrics.MeanAbsoluteError()]
        
        Learning_rate_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor = self.learn_rate_sched_factor,
            patience = self.learn_rate_sched_patience,  # Menší patience než u Early Stopping
            min_lr=0.0001 # Minimální rychlost učení
            )
        
        # Real-time plotting & early stopping callback
        live_plot_callback = LivePlotCallback(patience=self.patience)

        print("\nStarting training...\n")
        start_time = time.time()

        self.history = model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=[live_plot_callback,self.time_history, Learning_rate_scheduler],
            sample_weight=sample_weights
        )

        total_training_time = time.time() - start_time
        self.model = model
        self.Learning_rate_scheduler = Learning_rate_scheduler

        print("\nLearning is finished.")
        print(f"Total training time: {total_training_time:.2f} seconds.")

        return model

    def Evaluation(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call 'Model' method first.")

        # Evaluate the model on test data
        resonance_threshold = 1
        non_resonant_indices = self.y_test <= resonance_threshold
        X_test_filtered = self.X_test[non_resonant_indices]
        y_test_filtered = self.y_test[non_resonant_indices]

        loss = self.model.evaluate(X_test_filtered , y_test_filtered)
        self.loss = loss
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
        model_info['learning_rate_scheduler'] = self.Learning_rate_scheduler.__class__.__name__

        model_info['layers'] = []
        model_info['num_of_layers'] = 0
        model_info['neurons_per_layer'] = []  # This will store the list of neurons per layer
        model_info['activation_functions'] = []  # This will store the list of activation functions

        for layer in self.model.layers:
            layer_info = {}
            layer_info['layer_name'] = layer.__class__.__name__
            if isinstance(layer, keras.layers.Dense):
                # Get the number of neurons directly
                layer_info['num_neurons'] = layer.units
                # Get the activation function
                layer_info['activation_function'] = layer.get_config().get('activation', None)
                

                # Append the number of neurons and activation function to the respective lists
                model_info['neurons_per_layer'].append(layer.units)
                model_info['activation_functions'].append(layer.get_config().get('activation', None))

                # Increase the layer count
                model_info['num_of_layers'] += 1
            else:
                layer_info['num_neurons'] = None
                layer_info['activation_function'] = None

            # Add the layer info to the layers list
            model_info['layers'].append(layer_info)
            

        # Optimizer name
        model_info['optimizer'] = self.model.optimizer.__class__.__name__
        model_info['number_of_epochs'] = model_info['number_of_epochs'] = len(self.history.epoch) if hasattr(self.history, "epoch") else 0
        model_info['dataset_size'] = len(self.y)
        model_info['kernel_initializer'] = self.kernel_initializer
    # Calculate the mean of epoch_times (sum divided by length)
        epoch_times = self.time_history.get_epoch_times()    
        if epoch_times:  # Check if the list is not empty
            model_info['mean_epoch_time'] = sum(epoch_times) / len(epoch_times)
        else:
            model_info['mean_epoch_time'] = 0  # Default to 0 if the list is empty
        model_info['total_training_time'] = sum(epoch_times)  # Total time of training
        model_info['dropout_rate'] = self.dropout_rate
        model_info['mse_loss'] = self.loss
        model_info["regulaizers_values"] = [self.l1_value,self.l2_value]

        return model_info
    
