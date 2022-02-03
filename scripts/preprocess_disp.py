import os
import numpy as np
import mne
import osl
import yaml
from scipy.io import savemat


dataset_path = "/Users/ricsi/Documents/GitHub/MEG-transfer-decoding/scripts/rich_data/subj1/maxfiltered/"
output_directory = "/Users/ricsi/Documents/GitHub/MEG-transfer-decoding/scripts/rich_data/subj1/preproc25hz_epoched_meg"
decim = 10

config_text = """
meta:
    event_codes:
        words/hungry: 2
        words/tired: 3
        words/thirsty: 4
        words/toilet: 5
        words/pain: 6
preproc:
    - {method: filter, l_freq: 0.1, h_freq: 25}
    - {method: notch_filter, freqs: 50 100}
    - {method: bad_channels, picks: 'mag'}
    - {method: bad_channels, picks: 'grad'}
    - {method: bad_channels, picks: 'eeg'}
    - {method: bad_segments, segment_len: 800, picks: 'mag'}
    - {method: bad_segments, segment_len: 800, picks: 'grad'}
    - {method: find_events, min_duration: 0.002}
"""

"""
    - {method: ica_raw, picks: 'meg', n_components: 64}
    - {method: ica_autoreject, picks: 'meg', ecgmethod: 'correlation'}
"""

drop_log = open(os.path.join(output_directory, 'drop_log.txt'), 'w')
files = [f for f in os.listdir(dataset_path) if ('task' in f and 'mc.fif' in f)]
for f in files:
    raw = mne.io.read_raw_fif(os.path.join(dataset_path, f), preload=True)

    config = yaml.load(config_text, Loader=yaml.FullLoader)
    dataset = osl.preprocessing.run_proc_chain(raw, config)
    print(raw.info)
    raw = dataset['raw']

    if 'badchan' in output_directory:
        raw = raw.interpolate_bads()

    epochs = mne.Epochs(raw,
                        dataset['events'],
                        event_id=dataset['event_id'],
                        tmin=-0.4,
                        tmax=1.6,
                        baseline=None,
                        picks=['meg'],
                        reject={'grad': 0.0000000004, 'mag': 0.000000000008},
                        preload=True)

    #print(epochs.ch_names)

    for reason in epochs.drop_log:
        if reason:
            if reason[0] != 'IGNORED':
                print(reason)
                drop_log.write(str(reason) + '\n')

    for epoch, event in zip(epochs, epochs.events):
        data = epoch.T.astype(np.float32)
        event_id = event[-1]
        os.makedirs(f"{output_directory}/cond{event_id-2}", exist_ok=True)
        n_trials = int(len(os.listdir(f"{output_directory}/cond{event_id-2}"))/2)
        np.save(f"{output_directory}/cond{event_id-2}/trial{n_trials}.npy", data)
        savemat(f"{output_directory}/cond{event_id-2}/trial{n_trials}.mat", {'X': data})

drop_log.close()
