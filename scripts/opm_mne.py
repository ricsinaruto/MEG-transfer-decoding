import numpy as np
import os
import mne
import pickle
import pandas as pd
from mat73 import loadmat
import osl
import yaml


# Load raw data
raw = loadmat('../data/opm_rich/Task/raw.mat')
raw = raw['data']

# Select bad channels manually
drop_inds = [78, 79, 80, 30, 31, 32, 48, 49, 50, 63, 64, 65, 12, 13, 14, 3, 4, 5, 0, 1, 2, 6, 7, 8, 105, 106, 107]
good_inds = [i for i in range(186) if i not in drop_inds]

ch_names = [str(i) for i in good_inds]

# Drop bad channels
raw = raw[good_inds, :]
data = raw

# Load channel locations
channels = pd.read_csv('../data/opm_rich/Task/20220908_115229_channels.tsv', sep='\t')

# Set channel locations
chn_dict = []
for i in range(len(channels)):
    if i in good_inds:
        chn_positions = np.array([channels['Px'][i], channels['Py'][i], channels['Pz'][i], 0, 0, 0, 0, 0, 0, 0, 0, 0])

        chd = {'loc': chn_positions, 'ch_name': channels['name'][i], 'kind': 'FIFFV_EEG_CH'}
        chn_dict.append(chd)

# Create info structure
ch_names = [d['ch_name'] for d in chn_dict]
info = mne.create_info(ch_names=ch_names, sfreq=1200, ch_types='eeg')

# set channel locations
for i in range(len(chn_dict)):
    info['chs'][i]['loc'] = chn_dict[i]['loc']

# create mne raw object
raw = mne.io.RawArray(data, info)

raw.plot_sensors(show_names=True)

# save data
raw.save('../data/opm_rich/Task/raw.fif', overwrite=True)


# OSL preproc pipeline
outdir = os.path.join('..', 'data', 'opm_rich')

config_text = """
meta:
  event_codes:
    words/hungry: 2
    words/tired: 3
    words/thirsty: 4
    words/toilet: 5
    words/pain: 6
preproc:
  - filter:         {l_freq: 1, h_freq: 40}
  - notch_filter:   {freqs: 50 100}
  - bad_channels:   {picks: 'eeg'}
  - bad_segments:   {segment_len: 800, picks: 'eeg'}
  - ica_raw:        {picks: 'eeg', n_components: 50}
"""

# run OSL preprocessing
config = yaml.load(config_text, Loader=yaml.FullLoader)
dataset = osl.preprocessing.run_proc_chain('../data/opm_rich/Task/raw.fif', config, outdir=outdir, overwrite=True, gen_report=False)
