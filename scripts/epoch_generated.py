import os
import numpy as np

base = os.path.join('..', 'results', 'cichy_epoched', 'subj1', 'hmm', 'data_quant')

if '_100hz' not in base:
    path = os.path.join(base, 'event_times.npy')
    event_times = np.load(path)
    event_times = [(int(ev[0]/10) - 155520, ev[2]) for ev in event_times]
    event_times = [np.array(ev) for ev in event_times if ev[0] > 0]
    event_times = np.array(event_times)

else:
    path = os.path.join(base, 'generate_cond.npy')
    event_times = np.load(path)
    event_times = event_times[1:] - event_times[:-1]

    # get the indices where event_times is greater than 0
    evt = np.where(event_times > 0)[0]
    event_times = np.stack((evt, event_times[evt]), axis=1)

# load data
data = np.load(os.path.join(base, 'subject01.npy'))

# epoch data
epochs = []
for ev in event_times:
    epochs.append(data[ev[0]-10:ev[0]+100])

outdir = os.path.join(base, 'sub_epochs')

for epoch, event in zip(epochs, list(event_times[:, 1])):

    os.makedirs(f"{outdir}/cond{event-1}", exist_ok=True)
    n_trials = int(len(os.listdir(f"{outdir}/cond{event-1}")))
    np.save(f"{outdir}/cond{event-1}/trial{n_trials}.npy", epoch)
