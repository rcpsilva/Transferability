The H-score is a measure of transferability in the context of machine learning, particularly in tasks involving transfer learning. It essentially quantifies how well a source model or dataset can transfer its knowledge to a target model or dataset. The H-score is often used to compare the effectiveness of different pre-trained models when adapted to new tasks.

### Mathematical Definition of H-score:
The H-score for feature embeddings \( f \) is given by:
\[ H(f) = \text{tr}(\Sigma_f^{-1} \Sigma_z) \]
where:
- \( \Sigma_f \) is the covariance matrix of the feature embeddings.
- \( \Sigma_z \) is the covariance matrix of the conditional expectation of the features given the class labels.
- \( \text{tr} \) denotes the trace of a matrix, which is the sum of its diagonal elements.

### Intuition:
The H-score measures the discriminatory power of the features with respect to the classes. A higher H-score indicates that the feature space better separates the classes, which implies a higher potential for successful transfer learning. The measure captures both the spread (variance) of the features within each class and the distinctiveness between different classes.

### Calculation Example:
Suppose you have a dataset with 2 features and 3 samples, with the following values:

- Features for sample 1: \( [2, 3] \)
- Features for sample 2: \( [0, 1] \)
- Features for sample 3: \( [1, 3] \)

And suppose these samples belong to classes A, A, and B respectively. We first calculate the covariance matrix of the features, \( \Sigma_f \), and the covariance matrix of the features conditioned on the class labels, \( \Sigma_z \).

1. **Calculate the mean of features for each class:**
   - Mean for class A: \( [1, 2] \)
   - Mean for class B: \( [1, 3] \)

2. **Calculate \( \Sigma_f \) (overall covariance):**
   - Overall mean of features: \( [1, 7/3] \)
   - \( \Sigma_f = \frac{1}{2} \left( [(2-1)^2 + (0-1)^2 + (1-1)^2], [(3-7/3)^2 + (1-7/3)^2 + (3-7/3)^2] \right) \)

3. **Calculate \( \Sigma_z \) (class-conditional covariance):**
   - For class A: Variance of \( [2, 3] \) and \( [0, 1] \)
   - For class B: Variance of \( [1, 3] \)

4. **Compute H-score:**
   - Invert \( \Sigma_f \)
   - Compute \( \text{tr}(\Sigma_f^{-1} \Sigma_z) \)

These steps give a simplistic manual calculation of the H-score, though actual implementations would use libraries to handle matrix operations and more complex data structures.

```python
import numpy as np

def calculate_h_score(features, labels):
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

# Example data
features = np.array([[2, 3], [0, 1], [1, 3]])
labels = np.array(['A', 'A', 'B'])

# Calculate H-score
h_score = calculate_h_score(features, labels)
print("H-score:", h_score)
```