import numpy as np
from sklearn.feature_selection import mutual_info_classif

def interaction_info(X, y):
    """Mutual information between features and labels."""
    return np.mean(mutual_info_classif(X, y, discrete_features=True))