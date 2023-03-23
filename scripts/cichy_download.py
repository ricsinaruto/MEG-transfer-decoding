import os
import requests
import tarfile
import sys
from clint.textui import progress

from download import download_data


if not os.path.exists('data'):
    os.mkdir('data')

# check if there are any arguments
subs = 1
if len(sys.argv) > 1:
    subs = int(sys.argv[1])

# download data, number of subjects given by first script argument
for i in range(1, subs + 1):
    print('Downloading subject ', str(i))

    url = ('http://wednesday.csail.mit.edu/MEG2_MEG_Epoched_Raw_Data/Tars/' +
           f'subj{i:02d}_sess01.tar.gz')
    download_data(url, 'data/subj' + str(i-1) + '.tar.gz', 'data')
