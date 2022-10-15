import numpy as np
import os
import os.path as op
import mne
import pandas as pd


for subj in range(3, 5):
    print(f"Preprocessing subj{subj:02d}")
    

    # Input and output directories (for continuous data)
    raw_dir = f"/well/woolrich/projects/THINGS-MEG/raw_data/sub-BIGMEG{subj}"
    out_dir = f"/well/woolrich/projects/THINGS-MEG/preprocessed/25hz_lowpass_test/subj{subj-1}"

    os.makedirs(out_dir)

    for sess in range(1, 13):

        epoch_list = []
        # Load raw data for this subject and session
        for run in range(1, 11):
            path = op.join(raw_dir, f'ses-{sess:02d}', 'meg', f'sub-BIGMEG{subj}_ses-{sess:02d}_task-main_run-{run:02d}_meg.ds')
            raw = mne.io.read_raw_ctf(path, preload=True)

            raw.info['dev_head_t'] = None

            # Filters
            raw.filter(l_freq=1, h_freq=24.9, phase='minimum')

            # Read events file
            path = op.join(raw_dir, f'ses-{sess:02d}', 'meg', f'sub-BIGMEG{subj}_ses-{sess:02d}_task-main_run-{run:02d}_events.tsv')
            events = pd.read_csv(path, sep='\t')

            # apply lambda x: x > 0.1 to 'UADC016-2104' channel with mne
            raw.apply_function(lambda x: x > 1, verbose=True, picks=['UADC016-2104'])

            opt_events = mne.find_events(raw,
                                         stim_channel='UADC016-2104',
                                         initial_event=False,
                                         min_duration=0.1)

            events_mne = []
            for i in range(len(events)):
                if events['trial_type'][i] == 'test':
                    idd = int(events['test_image_nr'][i])
                    onset = opt_events[i, 0]
                    events_mne.append(np.array([onset, 0, idd]))

            length = len(events) - 1
            print(events['onset'][0], ' ', events['onset'][length])
            print(opt_events[0, 0]/1200, ' ', opt_events[length, 0]/1200)

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

            epoch_list.append(epochs)

        # Concatenate epochs
        epochs = mne.concatenate_epochs(epoch_list)

        # normalize across session
        scaler = mne.decoding.Scaler(scalings='mean')
        epochs_data = scaler.fit_transform(epochs.get_data())
        epochs_data = epochs_data.transpose(0, 2, 1).astype(np.float32)

        # Save epoched data
        for i, event in enumerate(epochs.events):
            data = epochs_data[i]
            event_id = event[-1]

            os.makedirs(f"{out_dir}/cond{event_id-1}", exist_ok=True)
            n_trials = len(os.listdir(f"{out_dir}/cond{event_id-1}"))
            np.save(f"{out_dir}/cond{event_id-1}/trial{n_trials}.npy", data)
