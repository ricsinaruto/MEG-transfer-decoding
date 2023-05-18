import os
import numpy as np
import mne
import osl
import yaml
from scipy.io import savemat


dataset_path = "/gpfs2/well/woolrich/projects/disp_csaky/CTF/11766/11766_Lukas_20230511_reading-f.ds"
outdir = "/gpfs2/well/woolrich/projects/disp_csaky/CTF/11766/sub_preproc25hz"

osl_outdir = os.path.join(outdir, 'oslpy')
report_dir = os.path.join(osl_outdir, 'report')
os.makedirs(report_dir, exist_ok=True)

event_dict = {'words/hungry': 2,
              'words/tired': 3,
              'words/thirsty': 4,
              'words/toilet': 5,
              'words/pain': 6}

config_text = """
meta:
    event_codes:
        words/hungry: 2
        words/tired: 3
        words/thirsty: 4
        words/toilet: 5
        words/pain: 6
preproc:
  - set_channel_types: {UADC007-4123: eog, UADC008-4123: eog, UADC009-4123: ecg}
  - filter:         {l_freq: 1, h_freq: 25, method: 'iir', iir_params: {order: 5, ftype: butter}}
  - resample:       {sfreq: 100}
  - bad_channels:   {picks: 'mag', ref_meg: False}
  - bad_segments:   {segment_len: 800, picks: 'mag', ref_meg: False}
  - ica_raw:        {picks: 'mag', n_components: 64}
  - ica_autoreject: {picks: 'mag', ecgmethod: 'correlation'}
"""

config = yaml.load(config_text, Loader=yaml.FullLoader)
dataset = osl.preprocessing.run_proc_chain(config,
                                           dataset_path,
                                           outdir=osl_outdir,
                                           overwrite=True,
                                           gen_report=True)

raw = dataset['raw']
# load preproc data directly from oslpy output
#raw = mne.io.read_raw_fif(os.path.join(osl_outdir, '13703054_Lukas_20230511_reading-f', '13703054_Lukas_20230511_reading-f_preproc_raw.fif'))

# drop bad channels
print(raw.info['bads'])
raw = raw.drop_channels(raw.info['bads'])

events = mne.events_from_annotations(raw)[0]

# only keep events with id below 7
events = events[events[:, -1] < 7]

epochs = mne.Epochs(raw,
                    events,
                    event_id=event_dict,
                    tmin=-0.1,
                    tmax=1.6,
                    baseline=None,
                    picks=['mag'],
                    reject=None,
                    preload=True,
                    event_repeated='drop')

print(len(epochs.ch_names))

for epoch, event in zip(epochs, epochs.events):
    data = epoch.T.astype(np.float32)

    event_id = event[-1]
    os.makedirs(f"{outdir}/cond{event_id-2}", exist_ok=True)
    n_trials = int(len(os.listdir(f"{outdir}/cond{event_id-2}")))
    if n_trials < 205:
        np.save(f"{outdir}/cond{event_id-2}/trial{n_trials}.npy", data)
