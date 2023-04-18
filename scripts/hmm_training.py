import os
os.environ["NVIDIA_VISIBLE_DEVICES"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from osl_dynamics import run_pipeline

savedir = os.path.join('..', 'results', 'cichy_epoched', 'subj1', 'hmm', 'gpt2_saved_kv_100hz')
loaddir = os.path.join(savedir, 'subject01.npy')

config = '''
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
      n_epochs: 20
      n_init: 3
      n_init_epochs: 1
'''

# run hmm
run_pipeline(config, loaddir, savedir)
