import os
import numpy as np
import mne
import osl
import yaml
from scipy.io import savemat


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
  - bad_segments:   {segment_len: 800, picks: 'eeg'}
  - find_events:    {min_duration: 0.002}
  - ica_raw:        {picks: 'eeg', n_components: 32}
  - ica_autoreject: {picks: 'eeg', ecgmethod: 'correlation', measure: 'correlation', threshold: 0.5}
"""

config_report = """
meta:
  event_codes:
preproc:
  - ica_raw:        {picks: 'meg', n_components: 64}
  - ica_autoreject: {picks: 'meg', ecgmethod: 'correlation'}
"""


for i in range(6):
    dataset_path = '/gpfs2/well/woolrich/projects/disp_csaky/s2/maxfiltered/subj' + str(i)
    files = [f for f in os.listdir(dataset_path) if ('task' in f and 'mc.fif' in f)]

    outdir = os.path.join(dataset_path, '..', '..', 'preproc25hz_eeg', 'subj' +str(i))
    os.makedirs(outdir, exist_ok=True)
    osl_outdir = os.path.join(outdir, '..', 'oslpy')
    report_dir = os.path.join(osl_outdir, 'report')
    os.makedirs(report_dir, exist_ok=True)

    raw = mne.io.read_raw_fif(os.path.join(dataset_path, files[0]), preload=True)

    config = yaml.load(config_text, Loader=yaml.FullLoader)
    dataset = osl.preprocessing.run_proc_chain(
        raw, config, outdir=osl_outdir, overwrite=True)

    print(raw.info)
    raw = dataset['raw']

    if 'badchan' in outdir:
        raw = raw.interpolate_bads()

    epochs = mne.Epochs(raw,
                        dataset['events'],
                        event_id=dataset['event_id'],
                        tmin=-0.4,
                        tmax=1.6,
                        baseline=None,
                        picks=['eeg'],
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

# generate report
osl_outdir = os.path.join('/gpfs2/well/woolrich/projects/disp_csaky/s2', 'preproc25hz_eeg', 'oslpy')
report_dir = os.path.join(osl_outdir, 'report')

files = [f for f in os.listdir(osl_outdir) if ('task' in f and 'raw.fif' in f)]
files = [os.path.join(osl_outdir, f) for f in files]

osl.report.gen_report(files, outdir=report_dir)
