import os
import requests
import tarfile
import sys
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

    # Remove the zipped file.
    os.remove(zipped_path)