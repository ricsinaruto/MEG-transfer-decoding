import os
import numpy as np
import mne
import osl
import yaml
from scipy.io import savemat


for i in [2, 3, 4]:
    sid = str(i)

    base = "/gpfs2/well/woolrich/projects/"
    dataset_path = base + f"disp_csaky/eeg/session{sid}/task.cdt"
    outdir = base + f"disp_csaky/eeg/session{sid}/preproc0.1_30hz"

    osl_outdir = os.path.join(outdir, 'oslpy')
    report_dir = os.path.join(osl_outdir, 'report')
    os.makedirs(report_dir, exist_ok=True)

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
    - filter:         {l_freq: 0.1, h_freq: 30}
    - bad_channels:   {picks: 'eeg'}
    - bad_segments:   {segment_len: 800, picks: 'eeg'}
    - ica_raw:        {picks: 'eeg', n_components: 32}
    - ica_autoreject: {picks: 'eeg', ecgmethod: 'correlation'}
    - find_events:    {stim_channel: 'Trigger', min_duration: 0.002}
    """

    drop_log = open(os.path.join(outdir, 'drop_log.txt'), 'w')

    raw = mne.io.read_raw_curry(dataset_path, preload=True)

    # apply this to the Trigger channel: ((x-0.061440)*1e6).astype(np.int32)+1
    fun = lambda x: ((x-0.061440)*1e6).astype(np.int32)+1
    raw.apply_function(fun, picks=['Trigger'])

    # set channel types
    raw.set_channel_types({'MAL': 'misc',
                           'MAR': 'misc',
                           'HEO': 'eog',
                           'VEO': 'eog',
                           'EKG': 'ecg',
                           'Trigger': 'stim'})

    config = yaml.load(config_text, Loader=yaml.FullLoader)
    dataset = osl.preprocessing.run_proc_chain(config,
                                               raw,
                                               outdir=osl_outdir,
                                               overwrite=True,
                                               outname='preproc',
                                               gen_report=True,
                                               reportdir=report_dir)

    print(dataset['raw'].info)

'''
epochs = mne.Epochs(dataset['raw'],
                    dataset['events'],
                    event_id=dataset['event_id'],
                    tmin=-0.4,
                    tmax=1.6,
                    baseline=None,
                    picks=['eeg'],
                    reject=None,
                    preload=True)

#print(epochs.ch_names)

for reason in epochs.drop_log:
    if reason:
        if reason[0] != 'IGNORED':
            print(reason)
            drop_log.write(str(reason) + '\n')

for epoch, event in zip(epochs, epochs.events):
    data = epoch.astype(np.float32)

    event_id = event[-1]
    os.makedirs(f"{outdir}/cond{event_id-2}", exist_ok=True)
    n_trials = int(len(os.listdir(f"{outdir}/cond{event_id-2}"))/2)
    np.save(f"{outdir}/cond{event_id-2}/trial{n_trials}.npy", data)
    savemat(f"{outdir}/cond{event_id-2}/trial{n_trials}.mat", {'X': data})
'''
