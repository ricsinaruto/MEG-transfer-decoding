import os
os.environ["NVIDIA_VISIBLE_DEVICES"] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from osl_dynamics import run_pipeline

savedir = os.path.join(
    '..', 'results', 'cichy_epoched', 'subj1', 'hmm', 'gpt2flat_group', 'p9')
loaddir = os.path.join(savedir, 'subject01.npy')
savedir = savedir

'''
    data_prep:
      n_embeddings: 15
      n_pca_components: 80
  hmm:
      n_states: 12
      sequence_length: 2000
      learn_means: False
      learn_covariances: True
      learn_trans_prob: True
      batch_size: 32
      learning_rate: 0.02
      n_epochs: 30
      n_init: 3
      n_init_epochs: 1
'''

config = '''
    load_data:
        inputs: ../results/cichy_epoched/subj1/hmm/gpt2flat_group/p9/subject01.npy
        kwargs:
            sampling_frequency: 100
            store_dir: tmp_9239
        prepare:
            tde_pca: {n_embeddings: 15, n_pca_components: 80}
            standardize: {}
    train_hmm:
        config_kwargs:
            n_states: 8
            sequence_length: 2000
            learn_means: False
            learn_covariances: True
            learn_trans_prob: True
            batch_size: 32
            learning_rate: 0.02
            n_epochs: 20
'''

'''
config = 
    load_data:
        inputs: ../results/cichy_epoched/shmm/data_quant/amplitude/subject01.npy
        kwargs:
            sampling_frequency: 100
            store_dir: tmp_9239
        prepare:
            amplitude_envelope: {}
            sliding_window: {n_window: 3}
            standardize: {}
    train_hmm:
        config_kwargs:
            n_states: 6
            sequence_length: 2000
            learn_means: True
            learn_covariances: True
            learn_trans_prob: True
            batch_size: 8
            learning_rate: 0.005
            n_epochs: 30

'''
# run hmm
run_pipeline(config, output_dir=savedir)
