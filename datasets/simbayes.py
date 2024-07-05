from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "simbayes"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'n_samples': [1000, 5000, 10000, 50000, 100000],
        'n_features': [5],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Generate pseudorandom data using `numpy`.
        rng = np.random.RandomState(42)  # same seed as in BayesEstimator utils
        beta = rng.randn(self.n_features)
        X = rng.randn(self.n_samples, self.n_features)
        s = X @ beta + rng.randn(self.n_samples)
        y = 2 * (s > 0).astype(int) - 1
        cat_indicator = [False]*X.shape[1]

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            X=X,
            y=y,
            categorical_indicator=cat_indicator
        )
