import torch
from main_kaggle_def import CharRNN

model_dante = 'rnn_20_epoch_sorted.net'

with open(model_dante, 'rb') as f:
    checkpoint = torch.load(f)

loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
loaded.load_state_dict(checkpoint['state_dict'])

print("loaded.chars = {}".format(loaded.chars))
print("loaded.int2char = {}".format(loaded.int2char))
print("loaded.char2int = {}".format(loaded.char2int))

for k, v in checkpoint['state_dict'].items():
    print("new {}".format(v.type()))
    print("key {}".format(k))
    print("size {}".format(v.size()))
    if len(v.size()) == 1:
        for c in v:
            print("{} ".format(c), end='')
        print()
    else:
        assert len(v.size()) == 2
        for r in v:
            for c in r:
                print("{} ".format(c), end='')
            print()
