import numpy as np
import os
import mne


inds = [154000, 48000, 85000, 70000, 57000, 67000, 49000, 89000, 187000, 56000, 37000, 106000, 81000, 83000, 240000]
for subj in range(3, 16):
    print(f"Preprocessing subj{subj:02d}")
    print("--------------------")

    # Directories
    raw_data_directory = f"/well/woolrich/projects/cichy118_cont/raw_data/subj{subj:02d}"
    output_directory = f"/well/woolrich/projects/cichy118_cont/preproc_data_onepass/cont/sub_folders/subj{subj-1}"

    #
    # Load raw data for this subject
    #
    raw = mne.io.read_raw_fif(f"{raw_data_directory}/MEG2_subj{subj:02d}_sess01_tsss_mc-0.fif")  # Loads all 4 files

    #
    # Apply some preprocessing before epoching
    #
    raw.load_data()

    events = mne.find_events(raw, min_duration=0.002)

    good_events = []
    for e in events:
        if e[2] < 119:
            e[0] -= inds[subj-1]
            good_events.append(e)

    good_events = np.array(good_events)

    #
    # Save data to file
    #
    np.save(f"{output_directory}/event_times.npy", good_events)
