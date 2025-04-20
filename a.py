import requests
import zipfile
import io
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split

def download_movielens():
    url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
    if not os.path.exists('ml-1m'):
        print("Downloading MovieLens 1M dataset...")
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
        print("Dataset downloaded and extracted.")

download_movielens()

ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python',
                      names=['user_id', 'item_id', 'rating', 'timestamp'])

user_ids = ratings['user_id'].unique()
item_ids = ratings['item_id'].unique()
user_map = {uid: idx for idx, uid in enumerate(user_ids)}
item_map = {iid: idx for idx, iid in enumerate(item_ids)}
ratings['user_idx'] = ratings['user_id'].map(user_map)
ratings['item_idx'] = ratings['item_id'].map(item_map)

num_users = len(user_ids)
num_items = len(item_ids)

train_data = []
test_data = []
for uid in user_ids:
    user_data = ratings[ratings['user_id'] == uid]
    train, test = train_test_split(user_data, test_size=0.2, random_state=42)
    train_data.append(train)
    test_data.append(test)

train_data = pd.concat(train_data)
test_data = pd.concat(test_data)

train_final, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

train_final['interaction'] = 1
R_train = csr_matrix((train_final['interaction'], (train_final['user_idx'], train_final['item_idx'])),
                     shape=(num_users, num_items))

user_degrees = np.array(R_train.sum(axis=1)).flatten()
item_degrees = np.array(R_train.sum(axis=0)).flatten()
D_U_inv_sqrt = diags(np.where(user_degrees > 0, 1.0 / np.sqrt(user_degrees), 0))
D_I_inv_sqrt = diags(np.where(item_degrees > 0, 1.0 / np.sqrt(item_degrees), 0))
R_hat = D_U_inv_sqrt @ R_train @ D_I_inv_sqrt

gamma = 0.04
m = int(gamma * min(num_users, num_items))

U, S, Vt = svds(R_hat, k=m)
S = S[::-1]
U = U[:, ::-1]
Vt = Vt[::-1, :]

Sigma_inv = diags(1.0 / S)

V_m = Vt.T
Q_m = U
Q_m_T_R = Q_m.T @ R_train
B = V_m @ (Sigma_inv @ Q_m_T_R)

R_tilde = R_hat @ B

def compute_metrics(R_tilde, test_data, train_data, K=10):
    hr = []
    ndcg = []
    for uid in test_data['user_idx'].unique():
        user_test = test_data[test_data['user_idx'] == uid]
        user_train = train_data[train_data['user_idx'] == uid]
        test_items = set(user_test['item_idx'])
        train_items = set(user_train['item_idx'])
        scores = R_tilde[uid]
        masked_scores = scores.copy()
        masked_scores[list(train_items)] = -np.inf
        top_k_indices = np.argsort(masked_scores)[::-1][:K]
        top_k_items = set(top_k_indices)
        if len(top_k_items & test_items) > 0:
            hr.append(1)
        else:
            hr.append(0)
        relevance = [1 if i in test_items else 0 for i in top_k_indices]
        ideal_relevance = sorted(relevance, reverse=True)
        dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance)])
        idcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance)])
        if idcg > 0:
            ndcg.append(dcg / idcg)
        else:
            ndcg.append(0)
    return np.mean(hr), np.mean(ndcg)

hr, ndcg = compute_metrics(R_tilde, test_data, train_data)
print(f"HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}")
