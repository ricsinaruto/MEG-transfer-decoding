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
  - filter:         {l_freq: 0.1, h_freq: 124.9}
  - notch_filter:   {freqs: '50'}
  - bad_channels:   {picks: 'mag'}
  - bad_channels:   {picks: 'grad'}
  - bad_channels:   {picks: 'eeg'}
  - bad_segments:   {segment_len: 800, picks: 'mag'}
  - bad_segments:   {segment_len: 800, picks: 'grad'}
  - find_events:    {min_duration: 0.002}
  - ica_raw:        {picks: 'meg', n_components: 64}
  - ica_autoreject: {picks: 'meg', ecgmethod: 'correlation'}
"""

config_report = """
meta:
  event_codes:
preproc:
  - ica_raw:        {picks: 'meg', n_components: 64}
  - ica_autoreject: {picks: 'meg', ecgmethod: 'correlation'}
"""


for i in range(15):
    dataset_path = '/gpfs2/well/woolrich/projects/disp_csaky/s1_raw_separate/subj' + str(i)
    files = [f for f in os.listdir(dataset_path) if ('task' in f and 'mc.fif' in f)]

    outdir = os.path.join(dataset_path, 'preproc125hz_ica_meg')
    osl_outdir = os.path.join(outdir, 'oslpy')
    os.makedirs(osl_outdir, exist_ok=True)

    raw = mne.io.read_raw_fif(os.path.join(dataset_path, files[0]), preload=True)

    config = yaml.load(config_text, Loader=yaml.FullLoader)
    dataset = osl.preprocessing.run_proc_chain(
        raw, config)

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
                        picks=['meg'],
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
'''
files = [f for f in os.listdir(osl_outdir) if ('task' in f and 'raw.fif' in f)]
files = [os.path.join(osl_outdir, f) for f in files]

osl.report.gen_report(files, outdir=report_dir, preproc_config=config_report)

drop_log.close()
'''
