import os
import numpy as np
import mne
import osl
import yaml
from scipy.io import savemat


dataset_path = "/gpfs2/well/woolrich/projects/disp_csaky/reading_only_meeg/ryan"
outdir = "/gpfs2/well/woolrich/projects/disp_csaky/reading_only_meeg/ryan/sub_preproc25hz"

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
  - filter:         {l_freq: 1, h_freq: 25, method: 'iir', iir_params: {order: 5, ftype: butter}}
  - resample:       {sfreq: 100}
  - bad_channels:   {picks: 'mag'}
  - bad_channels:   {picks: 'grad'}
  - bad_segments:   {segment_len: 800, picks: 'mag'}
  - bad_segments:   {segment_len: 800, picks: 'grad'}
  - ica_raw:        {picks: 'meg', n_components: 64}
  - ica_autoreject: {picks: 'meg', ecgmethod: 'correlation'}
  - find_events:    {min_duration: 0.002, shortest_event: 1}
"""
'''
config_text = """
meta:
    event_codes:
        words/hungry: 2
        words/tired: 3
        words/thirsty: 4
        words/toilet: 5
        words/pain: 6
preproc:
  - filter:         {l_freq: 1, h_freq: 25, method: 'iir', iir_params: {order: 5, ftype: butter}}
  - resample:       {sfreq: 100}
  - bad_channels:   {picks: 'eeg'}
  - bad_segments:   {segment_len: 800, picks: 'eeg'}
  - ica_raw:        {picks: 'eeg', n_components: 32}
  - ica_autoreject: {picks: 'eeg', ecgmethod: 'correlation'}
  - find_events:    {min_duration: 0.002, shortest_event: 1}
"""
'''

config_report = """
meta:
  event_codes:
preproc:
  - ica_raw:        {picks: 'eeg', n_components: 32}
  - ica_autoreject: {picks: 'eeg', ecgmethod: 'correlation'}
  - ica_raw:        {picks: 'meg', n_components: 64}
"""

drop_log = open(os.path.join(outdir, 'drop_log.txt'), 'w')
files = os.listdir(dataset_path)
#files = [f for f in files if 'mc.fif' in f]
files = [os.path.join(dataset_path, f'task_part{i}_rh_raw_tsss_mc.fif') for i in range(1, 4)]
raws = []
for f in files:
    raws.append(mne.io.read_raw_fif(os.path.join(dataset_path, f), preload=False))
    raws[-1].info['dev_head_t'] = None

raws = mne.concatenate_raws(raws, preload=True)

config = yaml.load(config_text, Loader=yaml.FullLoader)
dataset = osl.preprocessing.run_proc_chain(config, raws, outdir=osl_outdir, overwrite=True, gen_report=False)

print(raws.info)
raw = dataset['raw']

if 'badchan' in outdir:
    raw = raw.interpolate_bads()
else:
    raw = raw.drop_channels(raw.info['bads'])

epochs = mne.Epochs(raw,
                    dataset['events'],
                    event_id=dataset['event_id'],
                    tmin=-0.1,
                    tmax=1.6,
                    baseline=None,
                    picks=['meg'],
                    reject=None,
                    preload=True)

print(epochs.ch_names)

for reason in epochs.drop_log:
    if reason:
        if reason[0] != 'IGNORED':
            print(reason)
            drop_log.write(str(reason) + '\n')

drop_log.close()

for epoch, event in zip(epochs, epochs.events):
    data = epoch.astype(np.float32)

    event_id = event[-1]
    os.makedirs(f"{outdir}/cond{event_id-2}", exist_ok=True)
    n_trials = int(len(os.listdir(f"{outdir}/cond{event_id-2}")))
    if n_trials < 205:
        np.save(f"{outdir}/cond{event_id-2}/trial{n_trials}.npy", data)
