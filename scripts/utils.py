import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sentence_transformers import SentenceTransformer
import pytorch_lightning as L

class Movie100Dataset(Dataset):
    files = [
        'u.data',
        'u.item',
        'u.user',
        'u.genre',
        'u.occupation',        
    ]
    @staticmethod
    def assert_files(folder_path):
        for file in Movie100Dataset.files:
            assert os.path.exists(os.path.join(folder_path, file)), \
                f'File {file} not found in {folder_path}'
    
    def _load_user_repr(self, uuser):
        user_data = pd.read_csv(
            uuser,
            sep='|',
            names=['id', 'age', 'gender', 'occupation', 'zip_code']   
        )
        
        occupation = pd.get_dummies(user_data['occupation'])
        gender = pd.get_dummies(user_data['gender'])
        zip_hash = user_data.zip_code.apply(str).apply(hash).apply(lambda x: x >> 16).apply(abs)
        u = user_data.copy().drop(columns=['gender', 'occupation', 'zip_code'])
        u['zip_code'] = zip_hash
        u = u.merge(occupation, left_index=True, right_index=True)
        u = u.merge(gender, left_index=True, right_index=True)

        u = u.drop(columns=['id'])
        return torch.tensor(u.values.astype(np.float32), dtype=torch.float32)  
    
    def _load_item_repr(self, uitem):
        item_data = pd.read_csv(
            uitem,
            sep='|',
            encoding="unicode_escape",
            names=['id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'action', 'adventure', 'animation', 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']
        )
        
        embeddings = self.sentence_embedder.encode(
            item_data['title'].tolist(), convert_to_tensor=True
        )

        item_data['release_date'] = pd.to_datetime(item_data['release_date']).astype(int)
        item_data = item_data.drop(columns=['id', 'title', 'video_release_date', 'imdb_url'])
        
        vals = item_data.to_numpy()[:, 1:]

        vals = np.concatenate((vals, embeddings.cpu().numpy()), axis=1)
        
        return torch.tensor(vals, dtype=torch.float32)
    

    def __init__(
        self,
        sentence_embedder = None,
        folder_path = None,
        watch_threshold = 4,
    ):
        
        Movie100Dataset.assert_files(folder_path)
        
        self.folder_path = folder_path
        if sentence_embedder is None:
            self.sentence_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
        
        # (num_users, user_repr_dim)
        self.user_repr = self._load_user_repr(os.path.join(folder_path, 'u.user'))
        # (num_items, item_repr_dim)
        self.item_repr = self._load_item_repr(os.path.join(folder_path, 'u.item'))
        
        self.data = pd.read_csv(
            os.path.join(folder_path, 'u.data'),
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )
        
        self._user_ids = self.data.user_id.unique()
        self._item_ids = self.data.item_id.unique()
        self._watch_threshold = watch_threshold
        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user_id = row.user_id
        item_id = row.item_id
        rating = row.rating
        
        user = self.user_repr[user_id - 1]
        item = self.item_repr[item_id - 1]
        
        return user, item, rating
    
    @property
    def user_shape(self):
        return self.user_repr.shape[1]
    
    @property
    def item_shape(self):
        return self.item_repr.shape[1]
    
    
class Movie100DatasetNoText(Dataset):
        
    def _load_user_repr(self, uuser):
        user_data = pd.read_csv(
            uuser,
            sep='|',
            names=['id', 'age', 'gender', 'occupation', 'zip_code']   
        )
        
        occupation = pd.get_dummies(user_data['occupation'])
        gender = pd.get_dummies(user_data['gender'])
        zip_hash = user_data.zip_code.apply(str).apply(hash).apply(lambda x: x >> 16).apply(abs)
        u = user_data.copy().drop(columns=['gender', 'occupation', 'zip_code'])
        u['zip_code'] = zip_hash
        u = u.merge(occupation, left_index=True, right_index=True)
        u = u.merge(gender, left_index=True, right_index=True)

        u = u.drop(columns=['id'])
        return torch.tensor(u.values.astype(np.float32), dtype=torch.float32)  
    
    def _load_item_repr(self, uitem):
        item_data = pd.read_csv(
            uitem,
            sep='|',
            encoding="unicode_escape",
            names=['id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'action', 'adventure', 'animation', 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']
        )
        
        item_data['release_date'] = pd.to_datetime(item_data['release_date']).astype(int)
        item_data = item_data.drop(columns=['id', 'title', 'video_release_date', 'imdb_url'])
        
        return torch.tensor(item_data.to_numpy(), dtype=torch.float32)
    
    @property
    def user_shape(self):
        return self.user_repr.shape[1]
    
    @property
    def item_shape(self):
        return self.item_repr.shape[1]
    
    def __init__(
        self,
        folder_path = None,
    ):
        Movie100Dataset.assert_files(folder_path)
        
        self.folder_path = folder_path
        
        self.data = pd.read_csv(
            os.path.join(folder_path, 'u.data'),
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )
        
        self._user_ids = self.data.user_id.unique()
        self._item_ids = self.data.item_id.unique()
        
        self.user_repr = self._load_user_repr(os.path.join(folder_path, 'u.user'))
        self.item_repr = self._load_item_repr(os.path.join(folder_path, 'u.item'))
        

    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user_id = row.user_id
        item_id = row.item_id
        rating = row.rating
        
        user = self.user_repr[user_id - 1]
        item = self.item_repr[item_id - 1]
        
        return user, item, rating
    
    
    