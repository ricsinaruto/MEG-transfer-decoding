import numpy as np
import os
import mne

for subj in range(1, 2):
    print(f"Preprocessing subj{subj:02d}")
    print("--------------------")

    # Input and output directories (for continuous data)
    raw_data_directory = f"/well/woolrich/projects/cichy118_cont/raw_data/subj{subj:02d}"
    output_directory = f"/well/woolrich/projects/cichy118_cont/preproc_data_onepass/nofilter/cont_subj{subj-1}"
    if os.path.exists(output_directory):
        print("Please delete the following directory before running this script:")
        print(output_directory)
        exit()

    os.makedirs(output_directory)

    # Load raw data for this subject
    raw = mne.io.read_raw_fif(f"{raw_data_directory}/MEG2_subj{subj:02d}_sess01_tsss_mc-0.fif", preload=True)

    # Filters
    #raw.notch_filter(np.array([50, 100, 150]), phase='minimum')
    #raw.filter(l_freq=0.1, h_freq=124.9, phase='minimum')

    # Save continuous data
    data = raw.get_data(picks='meg')
    data = data.T.astype(np.float32)
    np.save(f'{output_directory}/data.npy', data)

    # Output directory for epoched data
    output_directory = f"/well/woolrich/projects/cichy118_cont/preproc_data_onepass/nofilter/subj{subj-1}"
    if os.path.exists(output_directory):
        print("Please delete the following directory before running this script:")
        print(output_directory)
        exit()

    # Extract epochs
    events = mne.find_events(raw, min_duration=0.002)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=list(range(1, 119)),
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
        os.makedirs(f"{output_directory}/cond{event_id-1}", exist_ok=True)
        n_trials = len(os.listdir(f"{output_directory}/cond{event_id-1}"))
        np.save(f"{output_directory}/cond{event_id-1}/trial{n_trials}.npy", data)
