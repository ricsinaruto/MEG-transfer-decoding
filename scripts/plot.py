import os
import mne

dataset_path = os.path.join('rich_data', 'subj1', 'preproc25hz_standardized_meg', 'oslpy',
                            'task_part2_6_raw_tsss_mc_raw.fif')

raw = mne.io.read_raw_fif(dataset_path, preload=True, verbose=True, on_split_missing='warn')
raw.pick_types(meg='mag')

raw.plot(start=20, n_channels=102, decim=19, clipping=2, scalings='auto', duration=300, block=True)