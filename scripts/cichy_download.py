import os
import requests
import tarfile
from clint.textui import progress


def download_data(url, zipped_path, extract):
    # Open the url and download the data with progress bars.

    data_stream = requests.get(url, stream=True)

    with open(zipped_path, 'wb') as file:
        total_length = int(data_stream.headers.get('content-length')) / 1024
        iter_stream = data_stream.iter_content(chunk_size=1024)
        for chunk in progress.bar(iter_stream, expected_size=total_length + 1):
            if chunk:
                file.write(chunk)
                file.flush()

    # Extract file.
    zip_file = tarfile.open(zipped_path, 'r:gz')
    zip_file.extractall(extract)
    zip_file.close()


if not os.path.exists('data'):
    os.mkdir('data')

for i in range(1, 16):
    print('Downloading subject ', str(i))

    url = ('http://wednesday.csail.mit.edu/MEG2_MEG_Epoched_Raw_Data/Tars/' +
           f'subj{i:02d}_sess01.tar.gz')
    download_data(url, 'data/subj' + str(i-1) + '.tar.gz', 'data')