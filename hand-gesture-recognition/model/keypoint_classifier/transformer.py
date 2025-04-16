import pandas as pd

class MaskFeatureSelector:
    def __init__(self):
        # Predefined mask for 42 landmarks
        self.mask = [False, False, True, True, False, False, False, False, False, False,
                     True, True, False, False, False, False, True, True, True, True, False, False,
                     False, False, True, True, True, True, False, False, False, False, False, False,
                     True, True, True, True, True, True, True, True]

    def fit(self, X, y=None):
        return self  # No fitting needed

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            # If X is a dataframe
            return X.iloc[:, self.mask]
        else:
            # If X is a numpy array
            return X[:, self.mask]  
