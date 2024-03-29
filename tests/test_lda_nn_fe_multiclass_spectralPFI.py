import os
import numpy as np

# load loss files
directory = os.path.join('results_test', 'lda_nn', 'subj0', 'spectralPFI')

path = os.path.join(directory, 'val_loss_PFIfreqs0.npy')
pfi = np.load(path)

# load true pfi
path = os.path.join('tests', 'results', 'lda_nn', 'subj0', 'spectralPFI')
path = os.path.join(path, 'val_loss_PFIfreqs0.npy')
true_pfi = np.load(path)

# assert that pfi is the same as true_pfi to 4 decimal places
# use numpy
assert np.allclose(pfi, true_pfi, rtol=0.02, atol=0.002)

# assert that args_saved.py is same as args.py
with open('args.py', 'r') as f:
    args = f.readlines()
with open(os.path.join(directory, 'args_saved.py'), 'r') as f:
    args_saved = f.readlines()

assert args == args_saved
