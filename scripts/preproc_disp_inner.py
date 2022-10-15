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

# NEED TO INTERPOLATE BAD CHANNELS

dataset_path = '/gpfs2/well/woolrich/projects/disp_csaky/RC/preproc250hz_deb'
outdir = '/gpfs2/well/woolrich/projects/disp_csaky/RC/preproc250hz_deb/inner_speech_sub_debchn2'
files = files = [f for f in os.listdir(dataset_path) if 'raw.fif' in f]
for f in files:
    raw = mne.io.read_raw_fif(os.path.join(dataset_path, f), preload=True)

    events = mne.find_events(raw, min_duration=0.002)
    event_c = np.array([e[2] for e in events])
    event_t = np.array([e[0] for e in events])

    #print(event_c[:20])
    count1 = 0
    count2 = 0
    new_events = []
    for i, (et, ec) in enumerate(zip(event_t, event_c)):
        if ec < 7 and ec > 1:
            count1 += 1
            if event_c[i+2] == 8:
                new_events.append(np.array([event_t[i+2], 0, ec]))
            else:
                print('error1')
            if event_c[i+3] == 8:
                new_events.append(np.array([event_t[i+3], 0, ec]))
            else:
                print('error2')
            if event_c[i+4] == 8:
                new_events.append(np.array([event_t[i+4], 0, ec]))
            else:
                print('error3')
            if event_c[i+5] == 8:
                new_events.append(np.array([event_t[i+5], 0, ec]))
            else:
                print('error4')

        elif ec < 16 and ec > 10:
            count2 += 1
            split_events = event_c[i-18:i-4]
            #print(split_events)
            tind = np.nonzero(split_events == 7)[0][-1]

            tind += i-18

            if event_c[tind+2] == 8:
                new_events.append(np.array([event_t[tind+2], 0, ec-9]))
            else:
                print('erorr5')
                print(event_c[i-20:i])
            if event_c[tind+3] == 8:
                new_events.append(np.array([event_t[tind+3], 0, ec-9]))
            else:
                print('erorr6')
                print(event_c[i-20:i])
            if event_c[tind+4] == 8:
                new_events.append(np.array([event_t[tind+4], 0, ec-9]))
            else:
                print('erorr7')
            if event_c[tind+5] == 8:
                new_events.append(np.array([event_t[tind+5], 0, ec-9]))
            else:
                print('erorr8')

    print(count1)
    print(count2)

    new_events = np.array(new_events)

    raw = raw.pick_channels(['MEG1213', 'MEG0612', 'MEG0423', 'MEG2322', 'MEG0232', 'MEG2612', 'MEG2532', 'MEG0722', 'MEG1143', 'MEG1022', 'MEG0513', 'MEG1212',
                             'MEG0743', 'MEG1723', 'MEG2012', 'MEG1613', 'MEG2042', 'MEG2523', 'MEG1832', 'MEG1543', 'MEG2333', 'MEG1443', 'MEG1943', 'MEG0533'])

    epochs = mne.Epochs(raw,
                        new_events,
                        event_id=event_dict,
                        tmin=-0.1,
                        tmax=1.0,
                        baseline=None,
                        picks=['grad'],
                        reject=None,
                        preload=True)

    epochs.drop_bad()

    #scaler = mne.decoding.Scaler(scalings='mean')
    #scaler.fit(epochs.get_data())
    for epoch, event in zip(epochs, epochs.events):
        data = epoch.T.astype(np.float32)
        #data = data.reshape(1, data.shape[0], -1)
        #data = scaler.transform(data).reshape(data.shape[1], -1).T

        event_id = event[-1]
        os.makedirs(f"{outdir}/cond{event_id-2}", exist_ok=True)
        n_trials = int(len(os.listdir(f"{outdir}/cond{event_id-2}"))/2)
        np.save(f"{outdir}/cond{event_id-2}/trial{n_trials}.npy", data)
        savemat(f"{outdir}/cond{event_id-2}/trial{n_trials}.mat", {'X': data})
