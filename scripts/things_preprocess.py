import numpy as np
import os
import os.path as op
import mne
import pandas as pd


for subj in range(1, 5):
    print(f"Preprocessing subj{subj:02d}")
    

    # Input and output directories (for continuous data)
    raw_dir = f"/well/woolrich/projects/THINGS-MEG/raw_data/sub-BIGMEG{subj}"
    out_dir = f"/well/woolrich/projects/THINGS-MEG/preprocessed/120hz_lowpass/subj{subj-1}"
    if os.path.exists(out_dir):
        print("Please delete the following directory before running this script:")
        print(out_dir)
        exit()

    os.makedirs(out_dir)

    for sess in range(1, 13):
        # Load raw data for this subject
        for run in range(1, 11):
            path = op.join(raw_dir, f'ses-{sess:02d}', 'meg', f'sub-BIGMEG{subj}_ses-{sess:02d}_task-main_run-{run:02d}_meg.ds')
            raw = mne.io.read_raw_ctf(path, preload=True)

            # Filters
            raw.filter(l_freq=0.1, h_freq=119.9, phase='minimum')

            # Read events file
            path = op.join(raw_dir, f'ses-{sess:02d}', 'meg', f'sub-BIGMEG{subj}_ses-{sess:02d}_task-main_run-{run:02d}_events.tsv')
            events = pd.read_csv(path, sep='\t')

            events_mne = []
            for i in range(len(events)):
                if events['trial_type'][i] == 'exp':
                    onset = int(events['onset'][i] * 1200)
                    idd = events['category_nr'][i]
                    events_mne.append(np.array([onset, 0 , idd]))

            events_mne = np.array(events_mne)

            # Extract epochs
            epochs = mne.Epochs(
                raw,
                events_mne,
                tmin=-0.1,
                tmax=1.0,
                baseline=None,
                picks="meg",
                preload=True
            )

            # Save epoched data
            for epoch, event in zip(epochs, epochs.events):
                data = epoch.T.astype(np.float32)
                event_id = event[-1]
                os.makedirs(f"{out_dir}/cond{event_id-1}", exist_ok=True)
                n_trials = len(os.listdir(f"{out_dir}/cond{event_id-1}"))
                np.save(f"{out_dir}/cond{event_id-1}/trial{n_trials}.npy", data)
