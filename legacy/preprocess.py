import os
import sys
import mne
import sails
from mat4py import loadmat
import numpy as np
import pickle

conditions = 118
trials = 30
sfreq = 1000
low_pass = 15
high_pass = 1
resample = 1


def pad_zeros(text, max_len):
    text = str(text)
    return (max_len - len(text)) * '0' + text


def preprocess():    
    '''
    The function does the following preprocessing steps after loading MEG data:
    1. Cropping
    2. Band-pass filtering
    3. Downsampling
    4. Bad channel removal
    5. ECG and EOG artefact removal using ICA
    '''

    data = []
    for i in range(1, conditions+1):
        trial_list = []
        for j in range(1, trials+1):
            trial = loadmat(os.path.join('data', 'subj01', 'cond0' + pad_zeros(i, 3), 'trial0' + pad_zeros(j, 2) + '.mat'))
            trial = np.array(trial['F'])
            trial_list.append(trial)
        data.append(np.array(trial_list))

    data = np.array(data)
    channels = data.shape[2]
    timesteps = data.shape[3]
    data = data.reshape(conditions * trials, channels, timesteps)

    info = mne.create_info(ch_names=channels, ch_types='mag', sfreq=sfreq)
    data = data.transpose(1, 0, 2).reshape(channels, conditions * trials * timesteps)
    raw = mne.io.RawArray(data, info)

    # Filter
    raw.filter(high_pass, low_pass)

    # Remove bad channels
    data = raw.get_data()
    good_inds = sails.utils.detect_artefacts(data, axis=0, ret_mode='good_inds')
    data = data[good_inds]
    channels = data.shape[0]

    data = data.reshape(channels, conditions * trials, timesteps).transpose(1, 0, 2)
    data = data[:, :, ::resample]
    pickle.dump(data, open(os.path.join('data', 'subj01', 'full_preprocessed_bandfiltered'), 'wb'))





if __name__ == "__main__":
    preprocess()