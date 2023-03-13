import numpy as np
import os
import mne
from scipy.io import loadmat

for subj in range(1, 16):
    sid = str(subj - 1)
    print('Preprocessing subj ', sid)

    raw_data = os.path.join(
        'data', 'scratch', 'rmcichy', 'FUB',
        'MEG2_long_baseline_MEG_Clean_Data_short_trials_from_long',
        'MEG_trials', f'subj{subj:02d}', 'sess01')
    output_directory = os.path.join('data', 'preproc', 'subj' + sid)
    os.makedirs(output_directory)

    epochs_mat = []
    event_id = []
    # Load data from 118 directories
    for c in range(1, 119):
        cond_path = os.path.join(raw_data, f'cond{c:04d}')

        for f in os.listdir(cond_path):
            trial = loadmat(os.path.join(cond_path, f))
            epochs_mat.append(trial['F'])

            event_id.append(c-1)

    epochs_mat = np.array(epochs_mat)

    # create epochs object
    channels = []
    for i in range(102):
        channels.extend(['mag', 'grad', 'grad'])
    info = mne.create_info(306, 1000, channels)

    epochs = mne.EpochsArray(np.array(epochs_mat), info)

    # Filters
    epochs.filter(l_freq=None, h_freq=25)

    # Save epoched data
    for epoch, event in zip(epochs, event_id):
        data = epoch.T.astype(np.float32)
        os.makedirs(f"{output_directory}/cond{event}", exist_ok=True)
        n_trials = len(os.listdir(f"{output_directory}/cond{event}"))
        np.save(f"{output_directory}/cond{event}/trial{n_trials}.npy", data)