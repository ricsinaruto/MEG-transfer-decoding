import os
import numpy as np
import mne
import pandas as pd

# reading trials
events = pd.read_csv(
    '../data/opm_rich/Task/20220908_115229_events.tsv', sep='\t')

event_c = np.array([events['value'][i] for i in range(len(events))])
event_t = np.array([events['sample'][i] for i in range(len(events))])

new_events = []
for ind, (et, ec) in enumerate(zip(event_t[1:-1], event_c[1:-1])):
    i = ind + 1

    plus = event_t[i+1] - et
    minus = et - event_t[i-1]
    if ec < 7 and ec > 1 and plus > 5 and minus > 5:
        new_events.append(np.array([et, 0, ec]))

new_events = np.array(new_events)

outdir = '/well/woolrich/projects/disp_csaky/opm_rich/reading/'
path = '/well/woolrich/projects/disp_csaky/opm_rich/raw_preproc_mfc.fif'
raw = mne.io.read_raw_fif(path, preload=True)


epochs = mne.Epochs(raw,
                    new_events,
                    tmin=-0.4,
                    tmax=1.6,
                    baseline=None,
                    picks=['eeg'],
                    reject=None,
                    preload=True)

epochs.drop_bad()

for epoch, event in zip(epochs, epochs.events):
    data = epoch.T.astype(np.float32)

    event_id = event[-1] - 2
    os.makedirs(f"{outdir}/cond{event_id}", exist_ok=True)

    n_trials = int(len(os.listdir(f"{outdir}/cond{event_id}")))
    np.save(f"{outdir}/cond{event_id}/trial{n_trials}.npy", data)
