import os
import argparse
import pandas as pd
import torch
from scripts.model import RecModelLightning, RecModelLightningSimpleFF
from scripts.utils import Movie100DatasetNoText

DIR = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(DIR, 'data')
BEST_MODEL = os.path.join(DIR, '..', 'models', 'best.ckpt')

data = pd.read_csv(
    os.path.join(DATA_FOLDER, 'u.data'), 
    sep='\t', 
    names=['user_id', 'item_id', 'rating', 'timestamp']
)

user = pd.read_csv(
    os.path.join(DATA_FOLDER, 'u.user'),
    sep='|',
    encoding="unicode_escape",
    names=['id', 'age', 'gender', 'occupation', 'zip_code']   
)

items = pd.read_csv(
    os.path.join(DATA_FOLDER, 'u.item'),
    sep='|',
    encoding="unicode_escape",
    names=['id', 'title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-noir', 'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western'],
)

# Initialize the dataset
ds = Movie100DatasetNoText(
    folder_path=DATA_FOLDER
)

# Initialize the model
model = RecModelLightningSimpleFF.load_from_checkpoint(BEST_MODEL)
model.eval()
model.to('cpu')


def predict(user_id, top_k=25):
    """Predicts the top K items for a given user.
    
    Args:
        user_id (int): The user ID.
        top_k (int): The number of items to predict.
        
    Returns:
        pd.DataFrame: The top K items.
    """
    user = ds.user_repr[user_id - 1]
    user = torch.tensor(user, dtype=torch.float32).unsqueeze(0)
    
    ratings = []
    for _, item in items.iterrows():
        item_id = item.id
        item = ds.item_repr[item_id - 1]
        item = torch.tensor(item, dtype=torch.float32).unsqueeze(0)
        rating, _, _ = model(user, item)
        ratings.append(rating.item())

    ratings = torch.tensor(ratings)
    top_k = ratings.topk(top_k)
    top_k = top_k.indices.tolist()
    
    return items.iloc[top_k]


def true_items(user_id, top_k=10):
    """Returns the top K items for a given user. 
    
    Args:
        user_id (int): The user ID.
        top_k (int): The number of items to return.
        
    Returns:
        pd.DataFrame: The top K items.
    """
    user_data = data[data.user_id == user_id]
    user_data = user_data[user_data.rating >= 4]
    user_data = user_data.sort_values(by='rating', ascending=False)
    return items.iloc[user_data.item_id[:top_k].tolist()]


def mean_average_precision(user_id, top_k=25):
    """Computes the mean average precision for a given user.
    
    Args:
        user_id (int): The user ID.
        top_k (int): The number of items to predict.    
        
    Returns:
        float: The mean average precision.
    """
    ti = true_items(user_id, top_k).id
    predicted_items = predict(user_id, top_k).id
    if len(predicted_items) == 0 or len(ti) == 0:
        return 0
    num_hits = 0
    total_precision = 0
    for i, item in enumerate(predicted_items):
        if item in ti:
            num_hits += 1
            total_precision += num_hits / (i + 1)
    
    return total_precision / len(ti)


def precision_at_k(user_id, top_k=25):
    """Computes the precision at K for a given user.
    
    Args:
        user_id (int): The user ID.
        top_k (int): The number of items to predict.
        
    Returns:
        float: The precision at K.
    """
    ti = true_items(user_id, top_k).id
    predicted_items = predict(user_id, top_k).id
    if len(predicted_items) == 0 or len(ti) == 0:
        return 0
    return len(set(predicted_items) & set(ti)) / len(predicted_items)


def recall_at_k(user_id, top_k=25):
    """Computes the recall at K for a given user.
    
    Args:
        user_id (int): The user ID.
        top_k (int): The number of items to predict.
        
    Returns:
        float: The recall at K.
    """
    ti = true_items(user_id, top_k).id
    predicted_items = predict(user_id, top_k).id
    if len(predicted_items) == 0 or len(ti) == 0:
        return 0
    
    return len(set(predicted_items) & set(ti)) / len(ti)

def evaluate(user_id, top_k=25):    
    """Evaluates the model for a given user.
    
    Args:
        user_id (int): The user ID.
        top_k (int): The number of items to predict.
    """
    ti = true_items(user_id, top_k).id
    predicted_items = predict(user_id, top_k).id
    print(sum([item in ti for item in predicted_items]))
    
    
def evaluate_all(top_k=25):
    """Evaluates the model for all users.
    
    Args:
        top_k (int): The number of items to predict.
    """    
    maps = []
    precisions = []
    recalls = []
    
    for user_id in data.user_id.unique():
        maps.append(mean_average_precision(user_id, top_k))
        precisions.append(precision_at_k(user_id, top_k))
        recalls.append(recall_at_k(user_id, top_k))
        
        print(f'User {user_id}')
        print(f'MAP: {maps[-1]}')
        print(f'Precision@K: {precisions[-1]}')
        print(f'Recall@K: {recalls[-1]}')
        print()

    pd.DataFrame({
        'user_id': ds._user_ids,
        'map': maps,
        'precision': precisions,
        'recall': recalls,
    }).to_csv('result.csv', index=False)
    
    print(f'MAP: {sum(maps) / len(maps) if len(maps) > 0 else 0}')
    print(f'Precision@K: {sum(precisions) / len(precisions) if len(precisions) > 0 else 0}')
    print(f'Recall@K: {sum(recalls) / len(recalls) if len(recalls) > 0 else 0}')
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user_id', type=int, default=None)
    
    args = parser.parse_args()
    
    if args.user_id is None:
        evaluate_all()
    else:
        evaluate(args.user_id)
        
    