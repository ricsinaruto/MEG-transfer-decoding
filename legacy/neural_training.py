import os
os.environ["NVIDIA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from mat73 import loadmat
from scipy.io import savemat
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mne.decoding import UnsupervisedSpatialFilter
import matplotlib.pyplot as plt
from matplotlib import colors
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from mne.time_frequency import tfr_array_morlet
import seaborn as sns
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from simpleffn import SimpleFFN
import sys

from vrad import data
from vrad.analysis import maps, spectral
from vrad.inference import metrics, states, tf_ops
from vrad.models import RIGO

from utils import load_data
from simpleffn import SimpleFFN
from simplecnn import Simple1DCNN

#tf_ops.gpu_growth()
multi_gpu = False

filtering = True
train_path = 'ryan_4states_newest'
remove_epochs = True
num_classes = 118
trials = 30
num_subjects = 1
pca_components = 80
resample = 4
tmin = 0
tmax = 1000
folds = 5
dropout = 0.2


learning_rate = 0.001
n_epochs = 500
mu = 127

# vrad params
n_states = 4
batch_size = 64
n_embeddings = 11

do_annealing = True
annealing_sharpness = 10
n_epochs_annealing = 300

rnn_type = "lstm"
rnn_normalization = "layer"
theta_normalization = "layer"

n_layers_inference = 1
n_layers_model = 1

n_units_inference = 64
n_units_model = 96

dropout_rate_inference = 0.0
dropout_rate_model = 0.0

learn_covariances = True

alpha_xform = "categorical"
alpha_temperature = 2.0
learn_alpha_scaling = False
normalize_covariances = False


def quantize(x_train, y_train):
    timeseries = x_train[:, 0, :]
    timeseries = timeseries - np.mean(timeseries)
    timeseries = timeseries / np.max(np.abs(timeseries))
    quantized = np.sign(timeseries)*np.log(1 + mu * np.abs(timeseries))/np.log(1 + mu)
    bins = np.linspace(-1, 1, mu + 1)
    quantized = np.digitize(quantized, bins) - 1

    #reconstructed = quantized + 1
    reconstructed = bins[quantized]
    reconstructed = (np.exp(np.abs(reconstructed) * np.log(1 + mu)) - 1)/mu*np.sign(reconstructed)

    plt.subplot(311)
    plt.plot(timeseries[10])
    plt.subplot(312)
    plt.plot(quantized[10])
    plt.subplot(313)
    plt.plot(reconstructed[10])
    plt.savefig(os.path.join('results', 'mu_quantize.svg'), format='svg', dpi=1200)
    plt.close('all')


def train(model_class, x_train, y_train):
    acc_all = []
    val_loss_all = []
    for fold in range(folds):
        # folds per class
        lower = int(fold*trials/folds)
        upper = int((fold+1)*trials/folds)
        x_val_all = torch.Tensor(np.array(np.concatenate(tuple([x_train[lower+trials*i:upper+trials*i] for i in range(num_classes)]))))
        x_train_all = torch.Tensor(np.array(np.concatenate(tuple([x_train[:lower]] + [x_train[upper+trials*i:lower+trials*(i+1)] for i in range(num_classes)]))))
        y_val_all = torch.tensor(np.array(np.concatenate(tuple([y_train[lower+trials*i:upper+trials*i] for i in range(num_classes)]))).reshape(-1), dtype=torch.long)
        y_train_all = torch.tensor(np.array(np.concatenate(tuple([y_train[:lower]] + [y_train[upper+trials*i:lower+trials*(i+1)] for i in range(num_classes)]))).reshape(-1), dtype=torch.long)

        model = model_class()

        # defining the optimizer
        optimizer = Adam(model.parameters(), lr=learning_rate)
        # defining the loss function
        criterion = CrossEntropyLoss()
        # checking if GPU is available
        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()

        # getting the training set
        x_train_all, y_train_all = Variable(x_train_all), Variable(y_train_all)
        x_val_all, y_val_all = Variable(x_val_all), Variable(y_val_all)

        # converting the data into GPU format
        if torch.cuda.is_available():
            x_train_all = x_train_all.cuda()
            y_train_all = y_train_all.cuda()
            x_val_all = x_val_all.cuda()
            y_val_all = y_val_all.cuda()


        accuracies = []
        val_losses = []
        train_losses = []
        # training the model
        for epoch in range(n_epochs):
            model.train()
            tr_loss = 0
            
            # clearing the Gradients of the model parameters
            optimizer.zero_grad()
            
            # prediction for training and validation set
            output_train = model(x_train_all)

            # computing the training and validation loss
            loss_train = criterion(output_train, y_train_all)

            # computing the updated weights of all the model parameters
            loss_train.backward()
            optimizer.step()


            model.eval()
            output_val = model(x_val_all)
            loss_val = criterion(output_val, y_val_all)
            output_val = np.argmax(output_val.detach().cpu().numpy(), axis=1)
            output_val = np.sum(np.equal(output_val,np.array(y_val_all.cpu())))/y_val_all.shape[0]
            if not epoch % 20:
                print('Epoch : ',epoch+1, '\t', 'train loss :', loss_train.item(), '\t', 'val loss: ', loss_val.item(),
                      '\t', 'val accuracy: ', output_val)

            accuracies.append(output_val)
            val_losses.append(loss_val.item())
            train_losses.append(loss_train.item())

        acc_all.append(accuracies)
        val_loss_all.append(val_losses)

    val_loss_avg = np.sum(np.array(val_loss_all), axis=0)/folds
    for loss in val_loss_all:
        plt.plot(loss)
    plt.savefig(os.path.join('results', 'losses.svg'), format='svg', dpi=1200)
    plt.close('all')

    acc_all = np.sum(np.array(acc_all), axis=0)/folds
    plt.plot(acc_all)
    plt.savefig(os.path.join('results', 'accs.svg'), format='svg', dpi=1200)
    plt.close('all')

    return acc_all, val_loss_avg


def cnn1d_training(x_train, y_train):
    Simple1DCNN.mean = np.mean(x_train)
    Simple1DCNN.std = np.std(x_train)

    Simple1DCNN.channels = x_train.shape[1]
    Simple1DCNN.num_classes = num_classes
    Simple1DCNN.dropout = dropout

    train(Simple1DCNN, x_train, y_train)


def ffn_training(x_train, y_train):
    x_train = x_train.reshape(num_classes*trials, -1)
    x_train = normalize(x_train)
    SimpleFFN.features = x_train.shape[1]
    SimpleFFN.classes = num_classes
    SimpleFFN.dropout = dropout

    train(SimpleFFN, x_train, y_train)


def vrad_training(x_train=None, y_train=None, num_examples=num_classes*trials, trial_list=[]):
    '''
    sequence_length = x_train.shape[2]
    num_channels = x_train.shape[1]
    num_trials = x_train.shape[0]
    raw = x_train.transpose(1, 0, 2)

    raw = raw.reshape(num_channels, num_examples * sequence_length).transpose()
    raw = (raw - np.mean(raw, axis=0))#/np.std(raw, axis=0)
    '''
    num_examples = 3533
    num_channels = pca_components
    sequence_length = 250
    raw = np.load('tmp/input_data_0_139617955146368.npy') * (1e10)

    raw_trials = raw.transpose().reshape(num_channels, num_examples, sequence_length)
    
    covariances = []
    for i in range(n_states):
        ''' # this starts with huge loss
        index = np.random.randint(num_trials)
        start = np.random.randint(sequence_length-101)
        matrix = raw[:, index, start+100]
        covariance = np.cov(matrix) + (1e-5) * np.eye(num_channels)
        covariances.append(covariance)
        '''

        indices = np.random.randint(num_examples, size=10)
        start = np.random.randint(sequence_length-41)
        matrix = raw_trials[:, indices, start:start+40].reshape(num_channels, -1)
        covariance = np.cov(matrix) + (1e-15) * np.eye(num_channels)
        covariances.append(covariance)


    covariances = np.array(covariances)

    #raw = raw.reshape(num_channels, num_examples * sequence_length).transpose()

    #savemat('raw_data_for_hmm.mat', {'X':raw})

    #prepared_data = data.PreprocessedData([raw])
    #prepared_data.prepare(n_pca_components=80, n_embeddings=n_embeddings, whiten=True, seq_len=sequence_length)
    #sequence_length = sequence_length - n_embeddings - 1
    prepared_data = data.Data([raw])
    # Prepare dataset
    training_dataset = prepared_data.training_dataset(sequence_length, batch_size)
    prediction_dataset = prepared_data.prediction_dataset(sequence_length, batch_size)



    # Build model
    model = RIGO(
        n_channels=80,
        n_states=n_states,
        sequence_length=sequence_length,
        learn_covariances=learn_covariances,
        rnn_type=rnn_type,
        rnn_normalization=rnn_normalization,
        n_layers_inference=n_layers_inference,
        n_layers_model=n_layers_model,
        n_units_inference=n_units_inference,
        n_units_model=n_units_model,
        dropout_rate_inference=dropout_rate_inference,
        dropout_rate_model=dropout_rate_model,
        theta_normalization=theta_normalization,
        alpha_xform=alpha_xform,
        alpha_temperature=alpha_temperature,
        learn_alpha_scaling=learn_alpha_scaling,
        normalize_covariances=normalize_covariances,
        do_annealing=do_annealing,
        annealing_sharpness=annealing_sharpness,
        n_epochs_annealing=n_epochs_annealing,
        learning_rate=learning_rate,
        multi_gpu=multi_gpu,
        initial_covariances=covariances
    )
    model.summary()

    print("Training model")
    history = model.fit(
        training_dataset,
        epochs=n_epochs,
        save_filepath='./results/vrad/S1_weights',
        save_best_after=10,
        use_tensorboard=True,
        tensorboard_dir='./results/vrad'
    )

    #pickle.dump(history, open('results/vrad/history', 'wb'))

    # Free energy = Log Likelihood + KL Divergence
    free_energy = model.free_energy(prediction_dataset)
    print(f"Free energy: {free_energy}")

    # Delete the temporary folder holding the data
    #prepared_data.delete_dir()

def plot_cov(covariance, name):
    for i, cov in enumerate(covariance):
        fig, ax = plt.subplots()
        cax = ax.matshow(cov, interpolation='nearest')
        ax.grid(True)
        fig.colorbar(cax)
        fig.savefig(os.path.join('results', 'vrad', train_path, name + str(i) + '.svg'), format='svg', dpi=1200)
        plt.close('all')


def test():
    num_examples = 3533
    num_channels = pca_components
    sequence_length = 250
    raw = np.load('tmp/input_data_0_139617955146368.npy') * (1e10)

    model = RIGO(
        n_channels=80,
        n_states=n_states,
        sequence_length=sequence_length,
        learn_covariances=learn_covariances,
        rnn_type=rnn_type,
        rnn_normalization=rnn_normalization,
        n_layers_inference=n_layers_inference,
        n_layers_model=n_layers_model,
        n_units_inference=n_units_inference,
        n_units_model=n_units_model,
        dropout_rate_inference=dropout_rate_inference,
        dropout_rate_model=dropout_rate_model,
        theta_normalization=theta_normalization,
        alpha_xform=alpha_xform,
        alpha_temperature=alpha_temperature,
        learn_alpha_scaling=learn_alpha_scaling,
        normalize_covariances=normalize_covariances,
        do_annealing=do_annealing,
        annealing_sharpness=annealing_sharpness,
        n_epochs_annealing=n_epochs_annealing,
        learning_rate=learning_rate,
        multi_gpu=multi_gpu,
    )
    model.summary()
    model.load_weights('./results/vrad/' + train_path + '/S1_weights')

    covz = model.get_covariances()
    plot_cov(covz, 'model')



    # static covariance
    static = np.cov(raw.transpose())
    plot_cov([static], 'static_cov')

    raw_trials = raw.transpose().reshape(num_channels, num_examples, sequence_length)
    covariances = []
    for i in range(n_states):
        indices = np.random.randint(num_examples, size=10)
        start = np.random.randint(sequence_length-41)
        matrix = raw_trials[:, indices, start:start+40].reshape(num_channels, -1)
        covariance = np.cov(matrix) + (1e-20) * np.eye(num_channels)
        covariances.append(covariance)
    plot_cov(covariances, 'init')

    prepared_data = data.Data([raw])

    # Prepare dataset
    prediction_dataset = prepared_data.prediction_dataset(sequence_length, batch_size)

    alpha = model.predict_states(prediction_dataset)[0]

    for i in range(n_states):
        plt.plot(alpha[:30*140, i])
    plt.savefig(os.path.join('results', 'vrad', train_path, 'alphas.svg'), format='svg', dpi=1200)
    plt.close('all')

    alpha_trials = alpha.reshape(num_examples, sequence_length, n_states)
    alpha_means = np.mean(alpha_trials, axis=0)
    for i in range(n_states):
        plt.plot(alpha_means[:,i])
    plt.savefig(os.path.join('results', 'vrad', train_path, 'alpha_means.svg'), format='svg', dpi=1200)
    plt.close('all')


    inf_stc = states.time_courses(alpha)
    #inf_stc = loadmat('gamma_cichy.mat')['gamma']
    # using argmax might not be the best approach
    inf_stc = np.argmax(inf_stc, axis=1)

    # Get states for each trial
    stc = inf_stc.reshape(num_examples, sequence_length)
    #stc = stc.reshape(num_classes, trials, sequence_length)

    cmap = colors.ListedColormap(['red', 'blue', 'yellow', 'green', 'pink', 'purple', 'brown', 'orange', 'cyan', 'white', 'lime', 'olive'])
    bounds = [i + 0.5 for i in range(n_states)]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    for i in range(num_classes):
        #print(stc[i][0])
        fig, ax = plt.subplots()
        lower = sum(trial_list[:i])
        upper = sum(trial_list[:i+1])
        ax.imshow(stc[lower:upper], cmap=cmap, norm=norm)
        #ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
        fig.savefig(os.path.join('results', 'vrad', train_path, 'stc', 'class' + str(i) + '.svg'), format='svg', dpi=1200)
        plt.close('all')

    # Inferred state mixing factors and state time courses
    inf_stc = states.time_courses(alpha)

    lifetimes = states.state_lifetimes(inf_stc)

    lifetimes = np.concatenate(tuple(lifetimes))
    print(len(lifetimes))
    print(sum(lifetimes))
    histogram = np.histogram(lifetimes, bins=100)
    with open('results/vrad/' + train_path + '/log.txt', 'w') as f:
        f.write(str(histogram))

    '''
    raw = raw.reshape(num_examples, sequence_length, num_channels)
    raw_mean = np.mean(raw, axis=0)

    for i in range(num_channels):
        plt.plot(raw_mean[:, i])
        plt.savefig(os.path.join('results', 'vrad', train_path, 'ch' + str(i) + '.svg'), format='svg', dpi=1200)
        plt.close('all')
    '''


def main():
    '''
    x_train, y_train, num_examples, trial_list = load_data(os.path.join('data', 'subj01', 'full_preprocessed'),
                                                 permute=False,
                                                 conditions=num_classes,
                                                 num_components=pca_components,
                                                 resample=resample,
                                                 tmin=tmin,
                                                 tmax=tmax,
                                                 remove_epochs=remove_epochs,
                                                 filtering=filtering)
    print(x_train.shape)
    '''

    #ffn_training(x_train, y_train)
    #cnn1d_training(x_train, y_train)
    vrad_training()
    #quantize(x_train, y_train)


if __name__ == "__main__":
    main()