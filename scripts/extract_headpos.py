import os
import numpy as np
import mne
import osl
import yaml
from scipy.io import savemat


outdir = "/gpfs2/well/woolrich/projects/disp_csaky/subj1/raw/"

files = [f for f in os.listdir(outdir) if ('task' in f and 'raw.fif' in f)]
for f in files:
    raw = mne.io.read_raw_fif(os.path.join(outdir, f), preload=True)

    chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
    chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=True)

    fname = os.path.join(outdir, f.split('raw.fif')[0] + 'headpos')
    mne.chpi.write_head_pos(fname, head_pos)
