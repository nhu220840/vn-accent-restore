import numpy as np

class GaussianNaiveBayes:
    """
    A Gaussian Naive Bayes classifier implemented from scratch.
    """
    
    def __init__(self):
        # We will store priors, means, and variances here
        self._classes = None
        self._n_classes = None
        self._priors = None
        self._means = None
        self._vars = None
        # Epsilon for numerical stability (to avoid division by zero)
        self._epsilon = 1e-9

    def fit(self, X, y):
        """
        Fit the GNB model to the training data.
        
        Parameters:
        X (np.ndarray): Training data of shape (n_samples, n_features)
        y (np.ndarray): Target labels of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        self._n_classes = len(self._classes)
        
        # Initialize arrays for stats
        # Shape: (n_classes, n_features)
        self._means = np.zeros((self._n_classes, n_features), dtype=np.float64)
        self._vars = np.zeros((self._n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(self._n_classes, dtype=np.float64)
        
        # Calculate stats for each class
        for idx, c in enumerate(self._classes):
            # Get all samples for the current class 'c'
            X_c = X[y == c]
            
            # Calculate mean, var, and prior
            self._means[idx, :] = X_c.mean(axis=0)
            self._vars[idx, :] = X_c.var(axis=0) + self._epsilon
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Predict class labels for a set of test data.
        
        Parameters:
        X (np.ndarray): Test data of shape (n_samples, n_features)
        
        Returns:
        np.ndarray: Predicted class labels
        """
        # Apply the _predict_sample helper function to each row in X
        y_pred = [self._predict_sample(x) for x in X]
        return np.array(y_pred)

    def _predict_sample(self, x):
        """Helper function to predict a single sample"""
        posteriors = []
        
        # Calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            # We use log probabilities for numerical stability
            log_prior = np.log(self._priors[idx])
            
            # Calculate log-likelihood for the sample 'x'
            # This is the sum of log-PDFs for each feature
            log_likelihood = np.sum(self._calculate_log_pdf(x, idx))
            
            # Log posterior = log prior + log likelihood
            log_posterior = log_prior + log_likelihood
            posteriors.append(log_posterior)
            
        # Return the class with the highest log posterior probability
        return self._classes[np.argmax(posteriors)]

    def _calculate_log_pdf(self, x, class_idx):
        """
        Calculates the log of the Gaussian Probability Density Function.
        log(PDF) = -0.5 * log(2*pi*var) - ((x - mean)^2 / (2*var))
        """
        mean = self._means[class_idx]
        var = self._vars[class_idx]
        
        term1 = -0.5 * np.log(2 * np.pi * var)
        term2 = -((x - mean) ** 2) / (2 * var)
        
        return term1 + term2