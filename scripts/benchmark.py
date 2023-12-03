import os
import shutil

if not os.path.exists('data/interim/val'):
    raise Exception('Please run scripts/split.py first!')

shutil.copytree('data/interim/val', 'benchmark/data')