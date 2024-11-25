import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD

def apply_nmf(user_item_matrix, n_components=20):
    nmf_model = NMF(n_components=n_components, init='random', random_state=42)
    user_features = nmf_model.fit_transform(user_item_matrix)
    item_features = nmf_model.components_
    reconstructed_matrix = np.dot(user_features, item_features)
    return user_features, item_features, reconstructed_matrix

def apply_svd(user_item_matrix, n_components=20):
    svd_model = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd_model.fit_transform(user_item_matrix)
    Vt = svd_model.components_
    reconstructed_matrix = np.dot(U, Vt)
    return U, Vt, reconstructed_matrix
