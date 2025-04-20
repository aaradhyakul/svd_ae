import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.utils.extmath import randomized_svd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import kagglehub


def load_data(file_path):
    """Load dataset and create user-item matrix"""
    df = pd.read_csv(file_path)
    # Convert user and item IDs to categorical indices
    users = df['user_id'].astype('category').cat.codes.values
    items = df['item_id'].astype('category').cat.codes.values
    
    num_users = len(df['user_id'].unique())
    num_items = len(df['item_id'].unique())
    
    # Create sparse interaction matrix
    R = csr_matrix((np.ones(len(df)), (users, items)), 
                    shape=(num_users, num_items))
    return R, num_users, num_items

def preprocess(R):
    """Normalize the interaction matrix"""
    # Compute degree matrices
    D_u = diags(1/np.sqrt(R.sum(axis=1).A.ravel()) 
    D_i = diags(1/np.sqrt(R.sum(axis=0).A.ravel())
    
    # Normalize R
    R_hat = D_u.dot(R).dot(D_i)
    return R_hat

def svd_ae(R_hat, gamma=0.04):
    """Compute truncated SVD and reconstruct matrix"""
    m = int(gamma * min(R_hat.shape))
    U, S, Vt = randomized_svd(R_hat, n_components=m, random_state=42)
    S_diag = np.diag(S)
    R_reconstructed = U @ S_diag @ Vt
    return R_reconstructed

def evaluate(test_users, test_items, scores, top_k=10):
    """Calculate HR@K and NDCG@K"""
    hr = 0
    ndcg = 0
    for u in range(len(test_users)):
        user_scores = scores[u]
        # Mask already seen items
        user_scores[test_users[u]] = -np.inf
        # Get top-k predictions
        top_items = np.argpartition(user_scores, -top_k)[-top_k:]
        # Check if test item is in top-k
        if test_items[u] in top_items:
            hr += 1
            rank = np.where(top_items == test_items[u])[0][0]
            ndcg += 1 / np.log2(rank + 2)
    hr /= len(test_users)
    ndcg /= len(test_users)
    return hr, ndcg

def main():
    # Load and split data
    dataset_path = kagglehub.dataset_download("odedgolden/movielens-1m-dataset")
    R, num_users, num_items = load_data(dataset_path)
    train_R, test_R = train_test_split(R, test_size=0.2, random_state=42)
    
    # Preprocess training data
    R_hat = preprocess(train_R)
    
    # Train SVD-AE
    reconstructed = svd_ae(R_hat)
    
    # Generate test pairs
    test_users, test_items = test_R.nonzero()
    
    # Evaluate
    hr, ndcg = evaluate(test_users, test_items, reconstructed, top_k=10)
    print(f"HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}")

if __name__ == "__main__":
    main()
