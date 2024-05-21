import numpy as np
import os
import mne
from scipy.io import loadmat

for subj in range(1, 23):
    print(f'Preprocessing subj{subj}')
    print("--------------------")

    # load .mat file
    name = f'LCIdata_lp25hz_{subj}.mat'
    mat = loadmat('/well/woolrich/projects/replay/study2_25hz/' + name)

    out_directory = f'/well/woolrich/projects/replay/study2_25hz/epoched/subj{subj}'

    epochs = np.array(mat['Cleandata_subset']).transpose(2, 0, 1)
    events = np.array(mat['stimlabel']).squeeze()
    goodchn = np.array(mat['goodchannindex']).squeeze()

    epochs = epochs[:, goodchn.astype(bool), :]
    print(epochs.shape)

    # Save epoched data
    for i, event in enumerate(events):
        data = epochs[i].astype(np.float32)
        os.makedirs(f"{out_directory}/cond{event-1}", exist_ok=True)
        n_trials = len(os.listdir(f"{out_directory}/cond{event-1}"))
        np.save(f"{out_directory}/cond{event-1}/trial{n_trials}.npy", data)
