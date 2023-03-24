import os
import requests
import tarfile
import sys
from clint.textui import progress

from download import download_data


if not os.path.exists('data'):
    os.mkdir('data')

if not os.path.exists('data/cichy92'):
    os.mkdir('data/cichy92')

# check if there are any arguments
subs = 1
if len(sys.argv) > 1:
    subs = int(sys.argv[1])

for i in range(1, subs+1):
    print('Downloading subject ', str(i))

    url = ('http://wednesday.csail.mit.edu/MEG1_MEG_Epoched_Raw_Data/Tars/' +
           f'subj{i:02d}_sess01.tar.gz')
    download_data(url, 'data/cichy92/subj' + str(i-1) + '_sess0.tar.gz', 'data/cichy92')

    url = ('http://wednesday.csail.mit.edu/MEG1_MEG_Epoched_Raw_Data/Tars/' +
           f'subj{i:02d}_sess02.tar.gz')
    download_data(url, 'data/cichy92/subj' + str(i-1) + '_sess1.tar.gz', 'data/cichy92')
