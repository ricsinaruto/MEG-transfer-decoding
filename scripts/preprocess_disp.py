import os
import numpy as np
import mne
import osl
import yaml
from scipy.io import savemat
from sklearn.preprocessing import StandardScaler


dataset_path = "/Users/ricsi/Documents/GitHub/MEG-transfer-decoding/scripts/rich_data/subj1/maxfiltered/"
output_directory = "/Users/ricsi/Documents/GitHub/MEG-transfer-decoding/scripts/rich_data/subj1/preproc25hz_standardized_meg"
osl_outdir = os.path.join(output_directory, 'oslpy')
os.makedirs(osl_outdir, exist_ok=True)
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
  - filter:         {l_freq: 0.1, h_freq: 25}
  - notch_filter:   {freqs: '50'}
  - bad_channels:   {picks: 'mag'}
  - bad_channels:   {picks: 'grad'}
  - bad_channels:   {picks: 'eeg'}
  - bad_segments:   {segment_len: 800, picks: 'mag'}
  - bad_segments:   {segment_len: 800, picks: 'grad'}
  - find_events:    {min_duration: 0.002}
"""

config_report = """
meta:
  event_codes:
preproc:
  - ica_raw:        {picks: 'meg', n_components: 64}
  - ica_autoreject: {picks: 'meg', ecgmethod: 'correlation'}
"""

drop_log = open(os.path.join(output_directory, 'drop_log.txt'), 'w')
files = [f for f in os.listdir(dataset_path) if ('task' in f and 'mc.fif' in f)]
for f in files:
    config = yaml.load(config_text, Loader=yaml.FullLoader)
    raw = mne.io.read_raw_fif(os.path.join(dataset_path, f), preload=True)
    dataset = osl.preprocessing.run_proc_chain(
        raw, config, outdir=osl_outdir, overwrite=True)

    print(raw.info)
    raw = dataset['raw']

    if 'badchan' in output_directory:
        raw = raw.interpolate_bads()

    '''

    epochs = mne.Epochs(raw,
                        dataset['events'],
                        event_id=dataset['event_id'],
                        tmin=-0.4,
                        tmax=1.6,
                        baseline=None,
                        picks=['meg'],
                        reject=None,
                        preload=True)

    #print(epochs.ch_names)

    for reason in epochs.drop_log:
        if reason:
            if reason[0] != 'IGNORED':
                print(reason)
                drop_log.write(str(reason) + '\n')

    scaler = mne.decoding.Scaler(scalings='mean')
    scaler.fit(epochs.get_data())
    for epoch, event in zip(epochs, epochs.events):
        data = epoch.astype(np.float32)
        data = data.reshape(1, data.shape[0], -1)
        data = scaler.transform(data).reshape(data.shape[1], -1).T

        event_id = event[-1]
        os.makedirs(f"{output_directory}/cond{event_id-2}", exist_ok=True)
        n_trials = int(len(os.listdir(f"{output_directory}/cond{event_id-2}"))/2)
        np.save(f"{output_directory}/cond{event_id-2}/trial{n_trials}.npy", data)
        savemat(f"{output_directory}/cond{event_id-2}/trial{n_trials}.mat", {'X': data})
    '''

# generate report
files = [f for f in os.listdir(osl_outdir) if ('task' in f and 'raw.fif' in f)]
files = [os.path.join(osl_outdir, f) for f in files]

report_dir = os.path.join(osl_outdir, 'report')
os.makedirs(report_dir, exist_ok=True)

osl.report.gen_report(files, outdir=report_dir, preproc_config=config_report)

drop_log.close()
