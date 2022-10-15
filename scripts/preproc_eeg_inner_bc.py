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

for i in [2, 3, 4, 5]:
    sid = str(i)

    fif_name = "preproc_preproc_raw.fif"
    base = "/gpfs2/well/woolrich/projects/disp_csaky/eeg/"
    dataset_path = base + f"session{sid}/preproc0.2_30hz/oslpy/" + fif_name
    outdir = base + "preproc0.2_30hz/inner_speech_longbc_3epochs/sub" + sid

    dataset = osl.preprocessing.read_dataset(dataset_path)

    events = dataset['events']

    event_c = np.array([e[2] for e in events])
    event_t = np.array([e[0] for e in events])

    count1 = 0
    count2 = 0
    new_events = []
    for i, (et, ec) in enumerate(zip(event_t, event_c)):
        if ec < 7 and ec > 1:
            count1 += 1
            if event_c[i+1] == 8:
                new_events.append(np.array([event_t[i+1], 0, ec]))
            else:
                print('error1')

        elif ec < 16 and ec > 10:
            count2 += 1
            split_events = event_c[i-18:i-4]
            #print(split_events)
            tind = np.nonzero(split_events == 7)[0][-1]

            tind += i-18

            if event_c[tind+1] == 8:
                new_events.append(np.array([event_t[tind+1], 0, ec-9]))
            else:
                print('erorr5')
                print(event_c[i-20:i])

    print(count1)
    print(count2)

    new_events = np.array(new_events)

    print(dataset['raw'].ch_names)

    raw = dataset['raw'].load_data()

    # filter HEO, VEO and EMG channels
    iir_params = {'order': 5, 'btype': 'bandpass', 'ftype': 'butter'}
    raw = raw.filter(0.2, 30,
                     picks=['HEO', 'VEO', 'EMG'],
                     method='iir',
                     iir_params=iir_params)

    epochs = mne.Epochs(raw,
                        new_events,
                        event_id=dataset['event_id'],
                        tmin=-1.0,
                        tmax=4.0,
                        baseline=(-0.1, 0),
                        picks=['eeg'],
                        reject=None,
                        preload=True)

    #scaler = mne.decoding.Scaler(scalings='mean')
    #scaler.fit(epochs.get_data())
    for epoch, event in zip(epochs, epochs.events):
        fulldata = epoch.T.astype(np.float32)
        event_id = event[-1]
        #data = data.reshape(1, data.shape[0], -1)
        #data = scaler.transform(data).reshape(data.shape[1], -1).T

        
        for i in range(1, 4):
            # select the data for the current time window (-1, 1)
            data = fulldata[i*1000:(i+2)*1000, :]

            os.makedirs(f"{outdir}/cond{event_id-2}", exist_ok=True)
            n_trials = int(len(os.listdir(f"{outdir}/cond{event_id-2}"))/2)
            np.save(f"{outdir}/cond{event_id-2}/trial{n_trials}.npy", data)
            savemat(f"{outdir}/cond{event_id-2}/trial{n_trials}.mat", {'X': data})