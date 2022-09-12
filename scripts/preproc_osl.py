import os
import numpy as np
import mne
import osl
import yaml
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

from osl.preprocessing.ica.plot_ica import plot_ica



dataset_paths = [os.path.join('rich_data', 'subj2', 'sess4', 'maxfilter_andrew', 'task_part1_4_raw_tsss.fif'),
                 os.path.join('rich_data', 'subj2', 'sess4', 'maxfilter_andrew', 'task_part2_4_raw_tsss.fif'),
                 os.path.join('rich_data', 'subj2', 'sess4', 'maxfilter_andrew', 'task_part3_4_raw_tsss.fif')]
outdir = os.path.join('rich_data', 'subj2', 'sess4', 'oslpy_deb_eeg')

config_text = """
meta:
  event_codes:
    words/hungry: 2
    words/tired: 3
    words/thirsty: 4
    words/toilet: 5
    words/pain: 6
    think: 7
    cue: 8
    twords/hungry: 11
    twords/tired: 12
    twords/thirsty: 13
    twords/toilet: 14
    twords/pain: 15
preproc:
  - filter:         {l_freq: 1, h_freq: 250}
  - notch_filter:   {freqs: 50 100 150 200 250 300}
  - bad_channels:   {picks: 'eeg'}
  - bad_segments:   {segment_len: 1000, picks: 'eeg'}
  - ica_raw:        {picks: 'eeg', n_components: 32}
  - ica_autoreject: {picks: 'eeg', ecgmethod: 'correlation', measure: 'correlation', threshold: 0.5, apply: False}
"""

raws = []
for d in dataset_paths:
    raws.append(mne.io.read_raw_fif(d, preload=False))
    print(raws[-1].info['dev_head_t'])
    raws[-1].info['dev_head_t'] = None
raws = mne.concatenate_raws(raws, preload=True)

config = yaml.load(config_text, Loader=yaml.FullLoader)
dataset = osl.preprocessing.run_proc_chain(raws, config, outdir=outdir, overwrite=True)


'''
dataset = {}
path = os.path.join(outdir, 'task_part1_rc_tsss_preproc_raw.fif')
dataset['raw'] = mne.io.read_raw_fif(path, preload=True)
path = os.path.join(outdir, 'task_part1_rc_raw_tsss_ica.fif')
dataset['ica'] = mne.preprocessing.read_ica(path)


plot_ica(dataset['ica'], dataset['raw'])
print(dataset['ica'].exclude)

dataset['raw'] = dataset['ica'].apply(dataset['raw'])

# drop channels
dataset['raw'] = dataset['raw'].drop_channels(dataset['raw'].info['bads'])

dataset['raw'].plot(duration=100, n_channels=100)

dataset['ica'].save(os.path.join(outdir, 'ica'))
dataset['raw'].save(os.path.join(outdir, 'raw'))
'''
