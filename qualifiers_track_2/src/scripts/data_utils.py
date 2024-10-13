import numpy as np
from nilearn.connectome import ConnectivityMeasure


def get_connectome(timeseries: np.ndarray,
                   conn_type: str = 'corr') -> np.ndarray:
    if conn_type == 'corr':
        conn = ConnectivityMeasure(kind='correlation', standardize=False).fit_transform(timeseries)
        conn[conn == 1] = 0.999999

        for i in conn:
            np.fill_diagonal(i, 0)

        conn = np.arctanh(conn)

    else:
        raise NotImplementedError

    return conn

def extract_features(X):
    """
    Function for extracting statistical characteristics from time series.
    X: an array of connectivity matrices
    Returns: new features (mean, median, standard deviation, variance for each brain region)
    """
    features = []
    for matrix in X:
        mean_features = np.mean(matrix, axis=1)
        median_features = np.median(matrix, axis=1)
        std_features = np.std(matrix, axis=1)
        var_features = np.var(matrix, axis=1)
        
        all_features = np.hstack([mean_features, median_features, std_features, var_features])
        features.append(all_features)
    
    return np.array(features)