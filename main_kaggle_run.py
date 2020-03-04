import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from main_kaggle_def import *

# open text file and read in data as `text`
with open('dante.txt', 'r') as f:
    text = f.read()

text[:300]

# encode the text and map each character to an integer and vice versa

# we create two dictionaries:
# 1. int2char, which maps integers to characters
# 2. char2int, which maps characters to unique integers
chars = sorted(tuple(set(text)))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

# encode the text
encoded = np.array([char2int[ch] for ch in text])

encoded[:100]

# check that the function works as expected
test_seq = np.array([[3, 5, 1]])
one_hot = one_hot_encode(test_seq, 8)

print(one_hot)

#when we call get batches we are going
#to create a generator that iteratest through our array and returns x, y with yield command

batches = get_batches(encoded, 8, 50)
x, y = next(batches)

# printing out the first 10 items in a sequence
print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])

# check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else:
    print('No GPU available, training on CPU; consider making n_epochs very small.')

# define and print the net
n_hidden=512
n_layers=4

net = CharRNN(chars, n_hidden, n_layers)
print(net)

batch_size = 64
seq_length = 160 #max length verses
n_epochs = 50 # start smaller if you are just testing initial behavior

# train the model
train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)

model_dante = 'rnn_20_epoch_sorted.net'

checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}

with open(model_dante, 'wb') as f:
    torch.save(checkpoint, f)

print(sample(net, 1000, prime='Nel ', top_k=5))
