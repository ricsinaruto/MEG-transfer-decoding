import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
import seaborn as sns
from osl_dynamics.utils import plotting
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
import faiss
from sklearn.metrics import pairwise_distances
from collections import Counter
import heapq

from osl_dynamics import run_pipeline, inference, analysis
from osl_dynamics.analysis import spectral
from osl_dynamics.data import Data, task


def balanced_init_medoids(distance_matrix, n_clusters):
    n_samples = distance_matrix.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    step = n_samples // n_clusters
    medoids = []

    for i in range(n_clusters):
        start = i * step
        end = start + step if i < n_clusters - 1 else n_samples
        medoid_idx = indices[start:end]
        medoids.append(medoid_idx)

    return np.array(medoids)



data_dir = os.path.join('/', 'well', 'woolrich', 'projects', 'cichy118_cont', 'preproc_data_osl', 'subj1', 'osl_dynamics100hz.mat')
data = loadmat(data_dir)['X'].T
data = data.astype(np.float32)
data = np.ascontiguousarray(data)
val_data = data[:10000]
data = data[10000:]

'''
# product quantization
# Set parameters for Product Quantization
num_subspaces = 1  # Number of subspaces
num_clusters_per_subspace = 8  # Total quantization bins = num_clusters_per_subspace ** num_subspaces = 100k

# Initialize the Product Quantizer index
quantizer = faiss.ResidualQuantizer(306, num_subspaces, num_clusters_per_subspace)
quantizer.train(data)

# encode
codes = quantizer.compute_codes(val_data)

# decode
recon = quantizer.decode(codes)

print(((val_data - recon)**2).sum() / (val_data ** 2).sum())

# count unique elements in each column of codes
for i in range(codes.shape[1]):
    print(np.unique(codes[:, i]).shape)
'''


# put the 306 channels into 6 buckets based on covariance
# in each bucket the channels should have similar covariances

num_buckets = 40
num_samples, num_features = data.shape

# Compute the covariance matrix of the features
cov_matrix = np.cov(data, rowvar=False)

# Apply K-means clustering on the covariance matrix
kmeans = KMeans(n_clusters=num_buckets, random_state=0).fit(cov_matrix)


# Create a dictionary to store the features in each bucket
buckets = {i: [] for i in range(num_buckets)}

# Assign features to the corresponding buckets
for feature_idx, bucket in enumerate(kmeans.labels_):
    buckets[bucket].append(feature_idx)

'''
# Create a dictionary to store the features in each bucket
buckets = {i: [] for i in range(num_buckets)}

# Initialize the cluster sizes
cluster_sizes = {i: 0 for i in range(num_buckets)}

# Assign features to the corresponding buckets
for feature_idx, bucket in enumerate(kmeans.labels_):
    buckets[bucket].append(feature_idx)
    cluster_sizes[bucket] += 1

# Iterate through the features and greedily reassign them
for feature_idx, initial_bucket in enumerate(kmeans.labels_):
    min_bucket = min(cluster_sizes, key=cluster_sizes.get)

    if cluster_sizes[min_bucket] < cluster_sizes[initial_bucket]:
        # Move the feature to the new cluster
        buckets[initial_bucket].remove(feature_idx)
        buckets[min_bucket].append(feature_idx)

        # Update the cluster sizes
        cluster_sizes[initial_bucket] -= 1
        cluster_sizes[min_bucket] += 1
'''


print(" ".join([str(len(buckets[i])) for i in buckets]))


num_subspaces = 2  # Number of subspaces
num_clusters_per_subspace = 6  # Total quantization bins = num_clusters_per_subspace ** num_subspaces = 100k
config = {'num_subspaces': [3], 'num_clusters_per_subspace': [6]}

unit = 600000//306
config = []
# define subspace and cluster sizes for each bucket based on bucket size
for bucket, features in buckets.items():
    size = unit * len(features)
    if size < 10000:
        config.append((2, 6))
    elif size < 24000:
        config.append((2, 7))
    elif size < 48000:
        config.append((3, 5))
    else:
        config.append((2, 8))

# create a residual quantizer for each bucket of features
errs = 0
vocabs = 0
residual_quantizers = {}
new_data = np.zeros(val_data.shape)
for c, (bucket, features) in zip(config, buckets.items()):
    tdata = np.ascontiguousarray(data[:, features])
    # optimize the number of subspaces and clusters per subspace
    '''
    best_quant = None
    best_err = 1000
    for i in len(config['num_subspaces']):
        ns = config['num_subspaces'][i]
        nc = config['num_clusters_per_subspace'][i]
        res_quant = faiss.ResidualQuantizer(len(features), ns, nc)
        res_quant.train(tdata)

        codes = res_quant.compute_codes(tdata)
        recon = res_quant.decode(codes)
        err = ((tdata - recon)**2).sum() / (tdata ** 2).sum()

        if err < best_err:
            best_quant = res_quant
            best_err = err
    '''
    #best_quant = faiss.ResidualQuantizer(len(features), c[0], c[1])
    #best_quant = faiss.LocalSearchQuantizer(len(features), num_subspaces, num_clusters_per_subspace)
    #best_quant.nperts = num_subspaces
    best_quant = faiss.ResidualQuantizer(len(features), num_subspaces, num_clusters_per_subspace)
    best_quant.train(tdata)

    vdata = np.ascontiguousarray(val_data[:, features])
    codes = best_quant.compute_codes(vdata)
    recon = best_quant.decode(codes)
    new_data[:, features] = recon

    # count unique elements in each column of codes
    vocab = 1
    for i in range(codes.shape[1]):
        vocab *= np.unique(codes[:, i]).shape[0]
    print(vocab, ' ', len(features))
    vocabs += vocab

    err = ((vdata - recon)**2).sum() / (vdata ** 2).sum()
    errs += err
    print(err)

print(vocabs)
err = ((val_data - new_data)**2).sum() / (val_data ** 2).sum()
print(err)