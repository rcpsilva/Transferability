import numpy as np

def h_score(features, labels):
    # Ensure the features are a NumPy array
    features = np.array(features)
    
    # Compute the overall mean of the features
    mean_overall = np.mean(features, axis=0)
    
    # Compute the overall covariance matrix Σ_f
    sigma_f = np.cov(features, rowvar=False)
    
    # Initialize the class-conditioned covariance matrix Σ_z
    sigma_z = np.zeros_like(sigma_f)
    
    # Calculate Σ_z as the weighted sum of class-specific covariance matrices
    classes = np.unique(labels)
    for c in classes:
        class_features = features[labels == c]
        class_mean = np.mean(class_features, axis=0)
        # Compute the covariance matrix for this class
        class_cov = np.cov(class_features, rowvar=False)
        # Weight by the proportion of this class in the dataset
        weight = len(class_features) / len(features)
        # Calculate the outer product of the mean difference and add to Σ_z
        mean_diff = class_mean - mean_overall
        sigma_z += weight * (class_cov + np.outer(mean_diff, mean_diff))
    
    # Calculate H-score as the trace of Σ_f^-1 * Σ_z
    sigma_f_inv = np.linalg.inv(sigma_f)
    h_score = np.trace(np.dot(sigma_f_inv, sigma_z))
    
    return h_score

import numpy as np

def shrinkage_based_h_score(features, labels, alpha):
    # Convert features to a numpy array
    features = np.array(features)
    
    # Calculate the overall mean of the features
    mean_overall = np.mean(features, axis=0)
    
    # Compute the empirical covariance matrix Σ_f
    sigma_f = np.cov(features, rowvar=False)
    
    # Calculate the class-specific means and overall class-conditional covariance Σ_z
    classes = np.unique(labels)
    sigma_z = np.zeros_like(sigma_f)
    for c in classes:
        class_features = features[labels == c]
        class_mean = np.mean(class_features, axis=0)
        class_cov = np.cov(class_features, rowvar=False)
        if class_features.shape[0] == 1:  # Handling for single sample per class
            class_cov = np.zeros_like(sigma_f)
        weight = len(class_features) / len(features)
        mean_diff = class_mean - mean_overall
        sigma_z += weight * (class_cov + np.outer(mean_diff, mean_diff))
    
    # Compute the shrinkage covariance matrix Σ_f^α
    average_variance = np.trace(sigma_f) / len(sigma_f)
    sigma_f_alpha = (1 - alpha) * sigma_f + alpha * average_variance * np.eye(len(sigma_f))
    
    # Compute the inverse of the shrinkage covariance matrix
    sigma_f_alpha_inv = np.linalg.inv(sigma_f_alpha)
    
    # Calculate the H-score
    h_score = np.trace(np.dot(sigma_f_alpha_inv, sigma_z))
    
    return h_score


if __name__ == '__main__':
    # Example data
    #features = np.array([[1, 4], [9, 16]])  # Squared features from class A and B
    #labels = np.array([0, 1])  # Class labels

    features = np.array([[2, 3], [0, 1], [1, 3]])
    labels = np.array(['A', 'A', 'B'])

    # Set the shrinkage coefficient
    alpha = 0.1

    # Calculate the shrinkage-based H-score
    sh_score = shrinkage_based_h_score(features, labels, alpha)
    h_score = h_score(features, labels)
    print("Shrinkage-based H-score:", sh_score)
    print("H-score:", h_score)