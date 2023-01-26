import os
import numpy as np
import mne
import osl
import yaml
import pickle
from scipy.io import savemat
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.animation as animation

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import euclidean_distances

# import distance_riemann
from pyriemann.utils.distance import distance_riemann

raws = []
for i in range(2, 12):
    sid = str(i)
    fif_name = "preproc_preproc_raw.fif"
    base = "/gpfs2/well/woolrich/projects/disp_csaky/eeg/"
    dataset_path = base + f"session{i}/preproc1_40hz_noica/oslpy/" + fif_name

    # load raw data
    raws.append(mne.io.read_raw_fif(dataset_path, preload=True))

# find epochs specific to words
event_id = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15}
epoch_event_id = {'words/hungry': 2, 'words/thirst': 3, 'words/tired': 4, 'words/toilet': 5, 'words/pain': 6}

epochs = []
for raw in raws:
    events = mne.events_from_annotations(raw, event_id=event_id)

    event_c = np.array([e[2] for e in events[0]])
    event_t = np.array([e[0] for e in events[0]])

    count1 = 0
    count2 = 0
    new_events = []
    for i, (et, ec) in enumerate(zip(event_t, event_c)):

        '''
        if ec < 7 and ec > 1:
            count1 += 1
            if event_c[i+1] == 8:
                new_events.append(np.array([event_t[i+1], 0, ec]))
            else:
                print('error1')
                continue

            if event_c[i+2] == 8:
                new_events.append(np.array([event_t[i+2], 0, ec]))
            else:
                print('error2')
            if event_c[i+3] == 8:
                new_events.append(np.array([event_t[i+3], 0, ec]))
            else:
                print('error3')
            if event_c[i+4] == 8:
                new_events.append(np.array([event_t[i+4], 0, ec]))
            else:
                print('error4')
        '''

        if ec < 16 and ec > 10:
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
            '''
            if event_c[tind+2] == 8:
                new_events.append(np.array([event_t[tind+2], 0, ec-9]))
            else:
                print('erorr6')
                print(event_c[i-20:i])
            if event_c[tind+3] == 8:
                new_events.append(np.array([event_t[tind+3], 0, ec-9]))
            else:
                print('erorr7')

            if event_c[tind+4] == 8:
                new_events.append(np.array([event_t[tind+4], 0, ec-9]))
            else:
                print('erorr8')
            '''
 




    print(count1)
    print(count2)

    new_events = np.array(new_events)

    ep = mne.Epochs(raw,
                    new_events,
                    event_id=epoch_event_id,
                    tmin=-0.5,
                    tmax=6,
                    baseline=None,
                    picks=['eeg'],
                    reject=None,
                    preload=True,
                    reject_by_annotation=False)
    epochs.append(ep)

'''
covs = []
for sess, ep in enumerate(epochs):
    data = ep.get_data()
    chn = data.shape[1]
    trials = data.shape[0]

    data = data.transpose(1, 2, 0).reshape(data.shape[1], -1)

    # standardize the data
    data = StandardScaler().fit_transform(data.T).T

    data = data.reshape(chn, -1, trials)

    trial_covs = []
    for i in range(trials):
        cov = np.cov(data[:, :, i])
        trial_covs.append(cov)

    sess_cov = np.mean(np.array(trial_covs), axis=0)

    outdir = base + f"preproc1_40hz_noica/thinkall_inner_speech4/cov{sess}.npy"
    np.save(outdir, sess_cov)
'''


for i, session in enumerate(epochs):
    outdir = base + f"preproc1_40hz_noica/thinkall_inner_speech_5.5s/sub" + str(i)
    for epoch, event in zip(session, session.events):
        data = epoch.T.astype(np.float32)

        event_id = event[-1]
        os.makedirs(f"{outdir}/cond{event_id-2}", exist_ok=True)
        n_trials = int(len(os.listdir(f"{outdir}/cond{event_id-2}"))/2)
        np.save(f"{outdir}/cond{event_id-2}/trial{n_trials}.npy", data)
        savemat(f"{outdir}/cond{event_id-2}/trial{n_trials}.mat", {'X': data})
