import numpy as np
from scipy.optimize import minimize
from numpy.linalg import inv, det, slogdet

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


def log_evidence(alpha, beta, F, y):
    D, n = F.shape[1], len(y)
    identity_matrix = np.eye(D)
    F_transpose = F.T

    # Compute the matrix A
    A = alpha * identity_matrix + beta * np.dot(F_transpose, F)
    
    # Compute the vector m
    m = beta * np.dot(inv(A), np.dot(F_transpose, y))
    
    # Compute the logarithm of the evidence
    _, logdet_A = slogdet(A)
    evidence = 0.5 * (n * np.log(beta) + D * np.log(alpha) - beta * np.dot(y - np.dot(F, m), y - np.dot(F, m)) - alpha * np.dot(m, m) - logdet_A)
    return -evidence  # Negative because we minimize in the optimizer

def logME(features, labels):
    # Initial guesses for alpha and beta
    initial_alpha = 1.0
    initial_beta = 1.0

    # Objective function to be minimized
    def objective(params):
        alpha, beta = params
        return log_evidence(alpha, beta, features, labels)
    
    # Minimization of the negative log evidence
    result = minimize(objective, [initial_alpha, initial_beta], bounds=((1e-5, None), (1e-5, None)))
    alpha_opt, beta_opt = result.x
    
    # Calculate the maximum log evidence using optimized parameters
    max_log_evidence = -log_evidence(alpha_opt, beta_opt, features, labels)
    
    return max_log_evidence


if __name__ == '__main__':
    # Example data
    #features = np.array([[1, 4], [9, 16]])  # Squared features from class A and B
    #labels = np.array([0, 1])  # Class labels

    features = np.array([[2, 3], [0, 1], [1, 3]])
    labels = np.array([1, 1, 0])

    # Set the shrinkage coefficient
    alpha = 0.1

    # Calculate the shrinkage-based H-score
    sh = shrinkage_based_h_score(features, labels, alpha)
    h = h_score(features, labels)
    lme = logME(features, labels)

    print("Shrinkage-based H-score:", sh)
    print("H-score:", h)
    print("LogME:", lme)