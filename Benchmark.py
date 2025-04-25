import numpy as np
import pandas as pd
import os

class NetworkBenchmark():
    def __init__(self, model_info):
        self.new_data = model_info
        self.K_Sequent_direct = r'D:\SKOLA\Vysoka skola\Inženýr\2.semestr\LDS\LDS_Strojove_uceni\Keras_Sequential_data.xlsx'
        self.RFR_direct = r'D:\SKOLA\Vysoka skola\Inženýr\2.semestr\LDS\LDS_Strojove_uceni\RFR_data.xlsx'


    def StoreData(self):
        # Check if model is Sequential
        if self.new_data['model_name'] == 'Sequential':
            if not os.path.exists(self.K_Sequent_direct):
                # If the file doesn't exist, create it with the proper columns
                df = pd.DataFrame(columns=["scaler_name", "num_of_layers", "neurons_per_layer", 
                                           "activation_functions", "optimizer","learning_rate_scheduler","kernel_initializer","dropout_rate","number_of_epochs","dataset_size", "mean_epoch_time",
                                            "total_training_time","mse_loss"])
                df.to_excel(self.K_Sequent_direct, index=False, engine='openpyxl')
                print("Excel file created.")
            else:
                # If file exists, load it
                df = pd.read_excel(self.K_Sequent_direct, engine='openpyxl')

            # Check for duplicates (ignoring epoch_times and mse_loss)
            duplicate_check = df[
                (df["scaler_name"] == self.new_data["scaler_name"]) &
                (df["num_of_layers"] == self.new_data["num_of_layers"]) &
                (df["neurons_per_layer"] == str(self.new_data["neurons_per_layer"])) &
                (df["activation_functions"] == str(self.new_data["activation_functions"])) &
                (df["optimizer"] == self.new_data["optimizer"]) &
                (df["learning_rate_scheduler"] == self.new_data["learning_rate_scheduler"]) &
                (df["kernel_initializer"] == self.new_data["kernel_initializer"]) &
                (df["dropout_rate"] == self.new_data["dropout_rate"]) &
                (df["dataset_size"] == self.new_data["dataset_size"]) &
                (df["number_of_epochs"] == self.new_data["number_of_epochs"])
            ]

            # Separate the first 3 rows (they will remain unchanged)
            if len(df) > 3:
                first_three_rows = df.iloc[:3]
                remaining_rows = df.iloc[3:]
            else:
                first_three_rows = df
                remaining_rows = pd.DataFrame(columns=df.columns)  # No data after the first three rows

            if duplicate_check.empty:
                # Append new data if no duplicates are found
                new_row = {
                    "scaler_name": self.new_data["scaler_name"],
                    "num_of_layers": self.new_data["num_of_layers"],
                    "neurons_per_layer": str(self.new_data["neurons_per_layer"]),  # Store as string
                    "activation_functions": str(self.new_data["activation_functions"]),  # Store as string
                    "optimizer": self.new_data["optimizer"],
                    "learning_rate_scheduler":self.new_data["learning_rate_scheduler"],
                    "kernel_initializer": self.new_data["kernel_initializer"],
                    "dropout_rate": self.new_data["dropout_rate"],
                    "number_of_epochs": self.new_data["number_of_epochs"],
                    "dataset_size": self.new_data["dataset_size"],
                    "mean_epoch_time": self.new_data["mean_epoch_time"],
                    "total_training_time": self.new_data["total_training_time"],
                    "mse_loss": self.new_data["mse_loss"]
                }
                updated_df = pd.concat([first_three_rows, pd.DataFrame([new_row]), remaining_rows], ignore_index=True)
        
                updated_df.to_excel(self.K_Sequent_direct, index=False, engine='openpyxl')
                print("New entry added to the Excel file.")
            else:
                print("Duplicate found, entry not added.")

        # For RandomForestRegressor model (optional, if you want to handle that as well)

        elif self.new_data['model_name'] == 'RandomForestRegressor':
            try:
                df = pd.read_excel(self.RFR_direct, engine='openpyxl')
            except FileNotFoundError:
                df = pd.DataFrame(columns=["scaler_name", "n_estimators", "max_depth", 
                                           "min_samples_split", "min_samples_leaf", "dataset_size",
                                           "training_time", "mse_loss"])

            # Check for duplicates based on model parameters
            duplicate_check = df[
                (df["scaler_name"] == self.new_data["scaler_name"]) &
                (df["n_estimators"] == self.new_data["n_estimators"]) &
                #(df["max_depth"] == self.new_data["max_depth"]) &
                (df["min_samples_split"] == self.new_data["min_samples_split"]) &
                (df["dataset_size"] == self.new_data["dataset_size"]) &
                (df["min_samples_leaf"] == self.new_data["min_samples_leaf"])
            ]

            # Separate the first 3 rows (they will remain unchanged)
            if len(df) > 3:
                first_three_rows = df.iloc[:3]
                remaining_rows = df.iloc[3:]
            else:
                first_three_rows = df
                remaining_rows = pd.DataFrame(columns=df.columns)  # No data after the first three rows

            if duplicate_check.empty:
                # Append new data if no duplicates are found
                new_row = {
                    "scaler_name": self.new_data["scaler_name"],
                    "n_estimators": self.new_data["n_estimators"],
                    "max_depth": self.new_data["max_depth"],
                    "min_samples_split": self.new_data["min_samples_split"],
                    "min_samples_leaf": self.new_data["min_samples_leaf"],
                    "dataset_size": self.new_data["dataset_size"],
                    "training_time": self.new_data["training_time"],
                    "mse_loss": self.new_data["mse_loss"]
                }
                updated_df = pd.concat([first_three_rows, pd.DataFrame([new_row]), remaining_rows], ignore_index=True)
                updated_df.to_excel(self.RFR_direct, index=False, engine='openpyxl')
                print("New entry added to the Excel file.")
            else:
                print("Duplicate found, entry not added.")

        else:
            print("Unknown model version")
