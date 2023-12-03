"""
Download and extract the MovieLens 100K dataset.
"""

import argparse
import requests
import zipfile

MOVIELENS_100K_URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/raw')
    parser.add_argument('--link', type=str, default=MOVIELENS_100K_URL)
    args = parser.parse_args()

    r = requests.get(args.link)
    with open('data/raw/ml-100k.zip', 'wb') as f:
        f.write(r.content)
        
    with zipfile.ZipFile('data/raw/ml-100k.zip', 'r') as zip_ref:
        zip_ref.extractall(args.data_dir)
        