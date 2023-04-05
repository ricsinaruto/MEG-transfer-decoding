import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
import seaborn as sns
from osl_dynamics.utils import plotting
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler

from osl_dynamics import run_pipeline, inference, analysis
from osl_dynamics import inference, analysis
from osl_dynamics.analysis import spectral
from osl_dynamics.data import Data, task
from osl_dynamics.models import load


def power_spectra(path):
    alp = pickle.load(open(os.path.join(path, 'inf_params', 'alp.pkl'), 'rb'))
    if os.path.exists(os.path.join(path, 'spectra')):
        # Load spectra
        f = np.load(os.path.join(path, 'spectra', 'f.npy'))
        psd = np.load(os.path.join(path, 'spectra', 'psd.npy'))
    else:
        # Compute spectra
        # Load data
        data = Data(path)

        # load model
        model = load(os.path.join(path, 'trained_model'))
        data = model.get_training_time_series(data, prepared=False)

        f, psd, coh = spectral.multitaper_spectra(data=data,
                                                  alpha=alp,
                                                  sampling_frequency=100,
                                                  time_half_bandwidth=4,
                                                  n_tapers=7,
                                                  frequency_range=[1, 50],
                                                  n_jobs=10)

        # Save spectra
        os.makedirs(os.path.join(path, 'spectra'), exist_ok=True)
        np.save(os.path.join(path, 'spectra', 'f.npy'), f)
        np.save(os.path.join(path, 'spectra', 'psd.npy'), psd)


path = os.path.join('..', 'results', 'cichy_epoched', 'subj1', 'hmm', '100hz')
power_spectra(path)

path = os.path.join('..', 'results', 'cichy_epoched', 'subj1', 'hmm', 'AR255_100hz')
power_spectra(path)

path = os.path.join('..', 'results', 'cichy_epoched', 'subj1', 'hmm', 'wavenet_100hz')
power_spectra(path)

path = os.path.join('..', 'results', 'cichy_epoched', 'subj1', 'hmm', 'gpt2_100hz')
power_spectra(path)
