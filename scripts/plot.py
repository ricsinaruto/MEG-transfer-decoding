import os
import mne
import osl

raw_path = os.path.join(
	'rich_data', 'subj1', 'preproc25hz_ica_meg', 'task_part4_5_raw_tsss_mc_raw.fif')

raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=True, on_split_missing='warn')
raw.pick_types(meg='mag')

raw.plot(start=20, n_channels=30, decim=19, clipping=2, duration=300, block=True)


#ica = mne.preprocessing.read_ica(raw_path.replace('raw.fif', 'ica.fif'))
#osl.preprocessing.osl_plot_ica.plot_ica(ica, raw)
