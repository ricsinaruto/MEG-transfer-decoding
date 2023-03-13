import numpy as np
import os
import mne
from scipy.io import loadmat

for subj in range(1, 17):
    sid = str(subj - 1)
    print('Preprocessing subj ', sid)
    output_directory = os.path.join('data', 'cichy92', 'preproc', 'subj' + sid)
    os.makedirs(output_directory)

    for sess in range(1, 3):
        raw_data = os.path.join(
            'data', 'cichy92', f'subj{subj:02d}', f'sess{sess:02d}')

        epochs_mat = []
        event_id = []
        # Load data from 118 directories
        for c in range(1, 93):
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