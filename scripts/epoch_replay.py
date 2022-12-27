import numpy as np
import os
from scipy.io import loadmat

for subj in range(1, 23):
    print(f'Preprocessing subj{subj}')
    print("--------------------")

    # load .mat file
    name = f'replaydata_study2_subj{subj}.mat'
    mat = loadmat('/well/woolrich/projects/replay/study2/' + name)

    out_directory = f'/well/woolrich/projects/replay/study2/epoched/subj{subj}'

    epochs = np.array(mat['lcidata']).transpose(2, 1, 0)
    events = np.array(mat['stimlabel']).squeeze()

    # Save epoched data
    for i, event in enumerate(events):
        data = epochs[i].astype(np.float32)
        os.makedirs(f"{out_directory}/cond{event-1}", exist_ok=True)
        n_trials = len(os.listdir(f"{out_directory}/cond{event-1}"))
        np.save(f"{out_directory}/cond{event-1}/trial{n_trials}.npy", data)
