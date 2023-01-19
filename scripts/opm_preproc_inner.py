import os
import numpy as np
import mne
import pandas as pd

# reading trials
events = pd.read_csv(
    '/well/woolrich/projects/disp_csaky/opm_lukas/20221019_085012_events.tsv', sep='\t')

event_c = np.array([events['value'][i] for i in range(len(events))])
event_t = np.array([events['sample'][i] for i in range(len(events))])

count1 = 0
count2 = 0
new_events = []
for ind, (et, ec) in enumerate(zip(event_t[1:-1], event_c[1:-1])):
    think_trial = False
    i = ind + 1

    plus = event_t[i+1] - et
    minus = et - event_t[i-1]

    if ec < 7 and ec > 1 and plus > 5 and minus > 5:
        if False:
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

    # elif
    elif (event_t[i+1] - et) < 5 and ec > 1:
        ec += event_c[i+1]
        think_trial = True

    if think_trial:
        count2 += 1
        split_events = event_c[i-18:i-4]

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

    # check if last two event times are the same
    if len(new_events) > 4:
        if new_events[-1][0] == new_events[-5][0]:
            print(new_events[-1][0])

        if new_events[-1][2] < 2:
            print(new_events[-1][2])
            print(new_events[-1][0])

print(count1)
print(count2)

new_events = np.array(new_events)

outdir = '/well/woolrich/projects/disp_csaky/opm_lukas/sub_innerspeech1_40hz_think'
path = '/well/woolrich/projects/disp_csaky/opm_lukas/raw_preproc_mark_mfc.fif'
raw = mne.io.read_raw_fif(path, preload=True)


epochs = mne.Epochs(raw,
                    new_events,
                    tmin=-1,
                    tmax=1,
                    baseline=None,
                    picks=['eeg'],
                    reject=None,
                    preload=True)

epochs.drop_bad()

for epoch, event in zip(epochs, epochs.events):
    data = epoch.T.astype(np.float32)

    event_id = event[-1] - 2
    if not os.path.exists(f"{outdir}/cond{event_id}"):
        os.makedirs(f"{outdir}/cond{event_id}")

    n_trials = int(len(os.listdir(f"{outdir}/cond{event_id}")))
    np.save(f"{outdir}/cond{event_id}/trial{n_trials}.npy", data)