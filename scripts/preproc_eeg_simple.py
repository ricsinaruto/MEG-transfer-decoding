import os
import numpy as np
import mne
import osl
import yaml
from scipy.io import savemat
import matplotlib.pyplot as plt


for i in range(2, 3):
    sid = str(i)

    base = "/gpfs2/well/woolrich/projects/"
    '''
    dataset_path = base + f"disp_csaky/eeg/session{sid}/task.cdt"
    outdir = base + f"disp_csaky/eeg/session{sid}/preproc1_20hz_noica"
    '''
    dataset_path = base + f"disp_csaky/eeg/lukas/task.cdt"
    outdir = base + f"disp_csaky/eeg/lukas/preproc1_40hz_noica_csd"

    osl_outdir = os.path.join(outdir, 'oslpy')
    report_dir = os.path.join(osl_outdir, 'report')
    os.makedirs(report_dir, exist_ok=True)

    txt = 'eeg'
    if 'csd' in outdir:
        txt = 'csd'

    config_text = """
    meta:
        event_codes:
            words/hungry: 2
            words/tired: 3
            words/thirsty: 4
            words/toilet: 5
            words/pain: 6
    preproc:
    - filter:         {l_freq: 1, h_freq: 40, method: 'iir', iir_params: {order: 5, ftype: 'butter'}}
    - bad_segments:   {segment_len: 200, picks: 'csd', significance_level: 0.1}
    - bad_segments:   {segment_len: 400, picks: 'csd', significance_level: 0.1}
    - bad_segments:   {segment_len: 600, picks: 'csd', significance_level: 0.1}
    - bad_segments:   {segment_len: 800, picks: 'csd', significance_level: 0.1}
    - find_events:    {stim_channel: 'Trigger', min_duration: 0.002}
    """
    '''
        - bad_channels:   {picks: 'eeg', significance_level: 0.4}
    '''

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

    if 'csd' in outdir:
        raw = mne.preprocessing.compute_current_source_density(raw)

    config = yaml.load(config_text, Loader=yaml.FullLoader)
    dataset = osl.preprocessing.run_proc_chain(config,
                                                raw,
                                                outdir=osl_outdir,
                                                overwrite=True,
                                                outname='preproc',
                                                gen_report=False)