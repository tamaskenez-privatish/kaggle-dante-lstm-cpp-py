import torch
import numpy as np
from main_kaggle_def import CharRNN, sample, one_hot_encode

model_dante = 'rnn_20_epoch_sorted.net'

with open(model_dante, 'rb') as f:
    checkpoint = torch.load(f)

loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
loaded.load_state_dict(checkpoint['state_dict'])

loaded.eval()

print('Loaded: {}', loaded)

if False:
    print('--- SAMPLING ---')
    print(sample(loaded, 100, prime='Nel '))
    print('--- SAMPLING DONE --- ')

x = np.array([[loaded.char2int['a']]])
x = one_hot_encode(x, len(loaded.chars))
inputs = torch.from_numpy(x)
h = loaded.init_hidden(1)

traced = torch.jit.trace(loaded, (inputs, h))

print('Traced: {}', traced)

if True:
    print('--- SAMPLING TRACED---')
    print(sample(loaded, 5, prime='Nel ', function=traced, top_k=1))
    print('--- SAMPLING DONE --- ')

#traced.save('traced.pt')
