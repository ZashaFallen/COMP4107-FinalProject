#External Libraries
import numpy as np
import tensorflow as tf

#Supplemetary Files
import data_prep as data

print("--- Dependancies Loaded ---")

data.process_data()
data.load_data()

print("--- Data Loaded ---")

#Hyperparameters
alpha = 0.001

#Begin working on model here