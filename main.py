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
epochs = 100
batch_size = 128

rnn_size = 512
layers = 2
encoding_size = 512
decoding_size = 512

alpha = 0.005
alpha_decay = 0.9
alpha_min = 0.0001
p_keep = 0.75

#Begin working on model here