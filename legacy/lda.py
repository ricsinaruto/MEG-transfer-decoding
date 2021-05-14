import os
import sys
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
import scipy
import matplotlib.pyplot as plt
from affinewarp import ShiftWarping, PiecewiseWarping

from utils import load_data


cross_val = 5
num_classes = 118
num_trials = 30
num_subjects = 1
pca_components = 50
resample = 7
tmin = 100
tmax = 800
alpha = np.arange(0.7, 1.3, 0.05)


def timewarped_distance(x, y):
    x = x.reshape(pca_components, -1)
    y = y.reshape(pca_components, -1)
    timesteps = y.shape[1]
    xp = np.linspace(0.0, 1.0, timesteps, endpoint=True)

    distances = []
    for a in alpha:
        yp = np.linspace(0.0, 1.0, int(timesteps * a), endpoint=True)
        resampled_y = np.array([np.interp(yp, xp, y[i]) for i in range(pca_components)])
        new_timesteps = min(timesteps, resampled_y.shape[1])

        dist = scipy.spatial.distance.euclidean(x[:, :new_timesteps].reshape(-1), resampled_y[:, :new_timesteps].reshape(-1))
        distances.append(dist)

    return min(distances)


def tsne():
    x_train, y_train = load_data(os.path.join('data', 'subj01', 'full_preprocessed'),
                                 permute=False,
                                 conditions=num_classes,
                                 num_components=pca_components,
                                 resample=resample,
                                 tmin=tmin,
                                 tmax=tmax)

    timesteps = x_train.shape[2]

    # time-warp data
    x_train_orig = x_train.reshape(num_classes, num_trials, pca_components, timesteps).transpose(0, 1, 3, 2)
    x_train = []
    for cond in range(num_classes):
        model = PiecewiseWarping(n_knots=1, warp_reg_scale=1e-6, smoothness_reg_scale=20.0)
        model.fit(x_train_orig[cond], iterations=50, warp_iterations=200)
        x_train.append(model.transform(x_train_orig[cond]))

    x_train = np.array(x_train).transpose(0, 1, 3, 2).reshape(num_classes * num_trials, pca_components, timesteps)
    x_train = x_train.reshape(num_classes*num_trials, -1)

    # analyze with tsne
    data_tsne = TSNE(metric=timewarped_distance).fit_transform(x_train)
    color_list = [np.random.rand(3) for i in range(num_classes)]

    fig, ax = plt.subplots()
    for i in range(num_classes*num_trials):
        ax.plot(data_tsne[i, 0], data_tsne[i, 1], 'o', color=color_list[y_train[i]], markersize=1)
    fig.savefig(os.path.join('results', 'image_cloud_timewarped.svg'), format='svg', dpi=1200)
    plt.close('all')


def lda():
    x_train, y_train = load_data(os.path.join('data', 'subj01', 'full_preprocessed'),
                                 permute=True,
                                 conditions=num_classes,
                                 num_components=pca_components,
                                 resample=resample,
                                 tmin=tmin,
                                 tmax=tmax)

    print(x_train.shape)
    x_train = x_train.reshape(num_classes*num_trials, -1)

    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    scores = cross_val_score(model, x_train, y_train, cv=cross_val)
    
    print(str(scores.mean()) + ' ' + str(scores.std()*2))
    return scores.mean()


if __name__ == "__main__":
    tsne()