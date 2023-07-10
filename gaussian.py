from scipy.stats import multivariate_normal

class GaussianGenerativeModel:
    def __init__(self):
        self.phi = None
        self.mu_0 = None
        self.mu_1 = None
        self.sigma = None

    def fit(self, X, y):
        # Compute phi, mu_0, mu_1, and sigma
        self.phi = sum(y) / len(y)
        self.mu_0 = X[y == 0].mean(axis=0)
        self.mu_1 = X[y == 1].mean(axis=0)
        self.sigma = ((X[y == 0] - self.mu_0).T @ (X[y == 0] - self.mu_0) + 
                      (X[y == 1] - self.mu_1).T @ (X[y == 1] - self.mu_1)) / len(y)

    def predict(self, X):
        # Compute the probability of each class for each data point in X
        p_0 = multivariate_normal.pdf(X, mean=self.mu_0, cov=self.sigma)
        p_1 = multivariate_normal.pdf(X, mean=self.mu_1, cov=self.sigma)
        
        # Assign the class with higher probability to each data point
        return (p_1 > p_0).astype(int)
