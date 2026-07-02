import numpy as np

class Normalizer:
    def __init__(self):
        self.params = {
            'min': None,
            'max': None
        }

    def fit(self, data):
        """Fit normalizer parameters to the data."""
        self.params['min'] = data[:, 0]
        self.params['max'] = data[:, 1]
    def transform(self, X):
        """Normalize data."""
        return (X - self.params['min']) / (self.params['max'] - self.params['min'])

    def inverse_transform(self, X_scaled):
        """Inverse the normalization to get back to the original scale."""
        return X_scaled * (self.params['max'] - self.params['min']) + self.params['min']
