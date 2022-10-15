import numpy as np
import os
import mne
from sklearn.preprocessing import StandardScaler


for subj in range(15):
    # load each subject with numpy
    data = np.load(f"/well/woolrich/projects/cichy118_cont/preproc_data_onepass/cont/subj{subj}")

    print(data.shape)

    data = StandardScaler().fit_transform(data)

    # compute covariance of data with numpy
    mat = np.triu(np.cov(data.T)).reshape(-1)
    cov = mat[mat != 0]

    # save covariance with numpy
    np.save(f"/well/woolrich/projects/cichy118_cont/preproc_data_onepass/cont/subj{subj}_cov", cov)

