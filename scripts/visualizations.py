import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import pickle
import seaborn as sns
import pandas as pd

# Load meg channel config
dataset_path = os.path.join('cichy_data', 'subj0', 'MEG2_subj01_sess01_tsss_mc-3.fif')
raw = mne.io.read_raw_fif(dataset_path, preload=True)
chn_type = 'mag'
raw = raw.pick(chn_type)

# load pfi data
path = os.path.join('..', 'results', 'cichy_epoched', 'all_noshuffle_wavenetclass_semb10_drop0.4',
                    'val_loss_PFIch4.npy')
pfi = np.load(open(path, 'rb'))
pfi = pfi[0, 0, 0] - pfi[:, :, 1:]

# times array
times = list(range(-48, 872, 4))
times = np.array([np.array(times) for _ in range(pfi.shape[0])])
times = np.array([times.reshape(-1) for _ in range(pfi.shape[2])]).T

# channels array
pfi_pd = pfi.reshape(-1, pfi.shape[2])
channels = np.array([np.arange(pfi_pd.shape[1]) for _ in range(pfi_pd.shape[0])])

# magnitudes for color hues
mags = np.abs(np.mean(pfi, axis=(0, 1)))
mags = np.array([1 - mags/np.max(mags) for _ in range(pfi_pd.shape[0])])

# put everything in a pd dataframe
pd_dict = {'pfi': pfi_pd.reshape(-1), 'times': times.reshape(-1), 'channels': mags.reshape(-1)}
pfi_pd = pd.DataFrame(pd_dict)

sns.relplot(
    data=pfi_pd, kind="line",
    x="times", y="pfi", hue="channels"
)