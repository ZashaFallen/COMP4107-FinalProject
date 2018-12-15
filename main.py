#Model adapted from: https://github.com/Currie32/Chatbot-from-Movie-Dialogue/blob/master/Chatbot_Attention.py

#External Libraries
import numpy as np
import tensorflow as tf

#Supplemetary Files
import data_prep as data

print("--- Dependancies Loaded ---")

data.process_data()
m_data, idx_q, idx_a = data.load_data()

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

#Creates the Encoder model
def encoder(input, n, l, p_k, s_len):
    lstm = tf.contrib.rnn.LSTMCell(n)
    dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = p_k)
    cell = tf.contrib.rnn.MultiRNNCell([dropout] * l)
    _, state = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell,
                                               cell_bw = cell,
                                               sequence_length = s_len,
                                               inputs = input, 
                                               dtype=tf.float32)
    return state
	
#Decodes the training data
def decoder_train(enc_state, d_cell, d_input, s_len, d_scope, o_fn, p_k, b_size):
	attention_states = tf.zeros([b_size, 1, d_cell.output_size])
	


	