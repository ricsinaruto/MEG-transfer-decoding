import os
import numpy as np
import mne
import osl
import yaml
from scipy.io import savemat


event_dict = {#'event_off':1,
              'words/hungry': 2,
              'words/tired': 3,
              'words/thirsty': 4,
              'words/toilet': 5,
              'words/pain': 6,
              #'think':7,
              #'cue':8,
              #'buttons_shown':9,
              #'error':10,
              #'twords/hungry':11,
              #'twords/tired':12,
              #'twords/thirsty':13,
              #'twords/toilet':14,
              #'twords/pain':15,
              #'button/1':256,
              #'button/2':512,
              #'button/3':1024,
              #'button/4':2048
             }


dataset_path = '/gpfs2/well/woolrich/projects/disp_csaky/s3/preproc40hz_eeg+meg_badchan/oslpy'
outdir = '/gpfs2/well/woolrich/projects/disp_csaky/s3/preproc40hz_eeg+meg_badchan/inner_speech_sub'
files = files = [f for f in os.listdir(dataset_path) if 'raw.fif' in f]
for f in files:
    raw = mne.io.read_raw_fif(os.path.join(dataset_path, f), preload=True)

    events = mne.find_events(raw, min_duration=0.002)

    new_events = []
    for i, e in enumerate(events):
        if e[2] == 8 and events[i-2][2] < 7:
            new_e = np.array([e[0], 0, events[i-2][2]])
            new_events.append(new_e)

    new_events = np.array(new_events)

    epochs = mne.Epochs(raw,
                        new_events,
                        event_id=event_dict,
                        tmin=-0.1,
                        tmax=1.6,
                        baseline=(None, 0),
                        picks=['eeg', 'meg'],
                        reject=None,
                        preload=True)

    scaler = mne.decoding.Scaler(scalings='mean')
    scaler.fit(epochs.get_data())
    for epoch, event in zip(epochs, epochs.events):
        data = epoch.astype(np.float32)
        data = data.reshape(1, data.shape[0], -1)
        data = scaler.transform(data).reshape(data.shape[1], -1).T

        event_id = event[-1]
        os.makedirs(f"{outdir}/cond{event_id-2}", exist_ok=True)
        n_trials = int(len(os.listdir(f"{outdir}/cond{event_id-2}"))/2)
        np.save(f"{outdir}/cond{event_id-2}/trial{n_trials}.npy", data)
        savemat(f"{outdir}/cond{event_id-2}/trial{n_trials}.mat", {'X': data})
