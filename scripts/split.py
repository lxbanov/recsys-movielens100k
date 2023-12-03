"""
Split the data into train and val sets.
"""


import pandas as pd
import os
import argparse
import shutil
SEED = 705

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/raw/ml-100k')
parser.add_argument('--output_dir', type=str, default='data/interim')

args = parser.parse_args()

data = pd.read_csv(
    os.path.join(args.data_dir, 'u.data'),
    sep='\t',
    names=['user_id', 'item_id', 'rating', 'timestamp']
)

user = pd.read_csv(
    os.path.join(args.data_dir, 'u.user'),
    sep='|',
    encoding="unicode_escape",
    names=['id', 'age', 'gender', 'occupation', 'zip_code']
)

item = pd.read_csv(
    os.path.join(args.data_dir, 'u.item'),
    sep='|',
    encoding="unicode_escape",
    names=['id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'action', 'adventure', 'animation', 'children', 'comedy',
           'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']
)

val_users = user.sample(frac=0.2, random_state=SEED).id
val_data = data[data.user_id.isin(val_users)]
# Create the output directory
os.makedirs(os.path.join(args.output_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)

# Creating the val dataset
val_data.to_csv(
    os.path.join(args.output_dir, 'val', 'u.data'),
    sep='\t', header=False, index=False
)

# Filter the user and item dataframes
user[user.id.isin(val_users)].to_csv(
    os.path.join(args.output_dir, 'val', 'u.user'),
    sep='|',
    header=False,
    index=False
)

item.to_csv(
    os.path.join(args.output_dir, 'val', 'u.item'),
    sep='|',
    header=False,
    index=False
)

# Copy the genre and occupation files (for data integrity)
shutil.copyfile(
    os.path.join(args.data_dir, 'u.genre'),
    os.path.join(args.output_dir, 'val', 'u.genre')
)

shutil.copyfile(
    os.path.join(args.data_dir, 'u.occupation'),
    os.path.join(args.output_dir, 'val', 'u.occupation')
)

# The same for the train dataset
user[user.id.isin(val_users)].to_csv(
    os.path.join(args.output_dir, 'val', 'u.split'),
    sep='|',
    header=False,
    index=False
)
shutil.copyfile(
    os.path.join(args.data_dir, 'u.user'),
    os.path.join(args.output_dir, 'val', 'u.user')
)

train_data = data[~data.user_id.isin(val_users)]
train_data.to_csv(
    os.path.join(args.output_dir, 'train', 'u.data'), 
    sep='\t', 
    header=False, 
    index=False
)

item.to_csv(
    os.path.join(args.output_dir, 'train', 'u.item'),
    sep='|',
    header=False, 
    index=False
)
shutil.copyfile(
    os.path.join(args.data_dir, 'u.genre'),
    os.path.join(args.output_dir, 'train', 'u.genre')
)
shutil.copyfile(
    os.path.join(args.data_dir, 'u.occupation'),
    os.path.join(args.output_dir, 'train', 'u.occupation')
)
shutil.copyfile(
    os.path.join(args.data_dir, 'u.user'),
    os.path.join(args.output_dir, 'train', 'u.user')
)
user[~user.id.isin(val_users)].to_csv(
    os.path.join(args.output_dir, 'train', 'u.split'), 
    sep='|', 
    header=False, 
    index=False
)
shutil.copyfile(
    os.path.join(args.data_dir, 'u.user'),
    os.path.join(args.output_dir, 'train', 'u.user')
)

print('Done!')
