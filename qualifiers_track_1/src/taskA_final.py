import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import pandas as pd
from scipy.stats import entropy

# Loading the data
raw_input = np.load('../data/ts_cut/ihb.npy')

# Print the shape of the data
print(f"Data shape: {raw_input.shape}")

# Find indices of rows containing NaN values
nan_row_indices = np.where(np.isnan(raw_input).any(axis=(1, 2)))[0]

# Get all row indices
total_indices = np.arange(320)
clean_row_indices = np.setdiff1d(total_indices, nan_row_indices)

# Split the data into subsets with and without NaN values
nan_data_subset = raw_input[nan_row_indices]
clean_data_subset = np.delete(raw_input, nan_row_indices, axis=0)

# Remove NaN values from the subset containing NaNs
cleaned_nan_data = np.array([arr[:, ~np.isnan(arr).any(axis=0)] for arr in nan_data_subset])


def extract_features(data_chunk):
    """Extract statistical features for each time series."""
    average_values = np.mean(data_chunk, axis=1)  # Mean
    deviation_values = np.std(data_chunk, axis=1)  # Standard deviation
    min_values = np.min(data_chunk, axis=1)  # Minimum values
    max_values = np.max(data_chunk, axis=1)  # Maximum values
    variance_values = np.var(data_chunk, axis=1)  # Variance
    entropy_values = np.apply_along_axis(lambda x: entropy(np.histogram(x, bins=10)[0] + 1), 1, data_chunk)  # Entropy

    # Combine all extracted features into a single array
    all_features = np.hstack((average_values, deviation_values, min_values, max_values, variance_values, entropy_values))

    # Normalize the features
    feature_scaler = StandardScaler()
    normalized_features = feature_scaler.fit_transform(all_features)

    # Apply PCA for dimensionality reduction (preserving 95% variance)
    pca_processor = PCA(n_components=0.95)
    reduced_features = pca_processor.fit_transform(normalized_features)

    return reduced_features


def apply_clustering(reduced_features):
    """Apply clustering on reduced features."""
    clustering_model = SpectralClustering(n_clusters=25, random_state=75, affinity='nearest_neighbors')
    labels = clustering_model.fit_predict(reduced_features)

    return labels


# Get cluster labels for both subsets of data
nan_features = extract_features(cleaned_nan_data)
clean_features = extract_features(clean_data_subset)

nan_cluster_labels = apply_clustering(nan_features)
clean_cluster_labels = apply_clustering(clean_features)

# Shift the labels for the second dataset to create 50 unique classes
shifted_clean_labels = clean_cluster_labels + 25

# Create an empty array to store final cluster labels for all 320 rows
final_cluster_labels = np.empty(len(nan_row_indices) + len(clean_row_indices), dtype=int)

# Assign labels to rows with NaN values
final_cluster_labels[nan_row_indices] = nan_cluster_labels

# Assign labels to rows without NaN values
final_cluster_labels[clean_row_indices] = shifted_clean_labels

# Save the final cluster labels to a CSV file
pd.DataFrame({'prediction': final_cluster_labels}).to_csv('../submission7777.csv', index=False)

print('Clustering completed, results saved to "submission7777.csv".')
