from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class BayesEstimator(BaseSolver):
    def __init__(self, noise):
        # The following seed must be the same as in the dataset simbayes.py
        self.rng = np.random.RandomState(42)
        self.beta = None  # Placeholder for beta
        self.model = lambda X: self._model_helper(X, noise)

    def _model_helper(self, X, noise):
        if self.beta is None:
            self.beta = self.rng.randn(X.shape[1])
        return X @ self.beta + noise * self.rng.randn(X.shape[0]) > 0

    def fit(self, X, y):
        pass

    def score(self, X, y):
        return np.mean(self.model(X) == y)

    def predict(self, X):
        return self.model(X)

    def predict_proba(self, X):
        # Compute the probability estimates for each class.
        # In this case, we can use the sign of the model's output as the
        # probability estimate.
        predictions = self.model(X)
        proba = np.zeros((X.shape[0], 2))
        proba[:, 0] = (predictions == -1)
        proba[:, 1] = (predictions == 1)
        return proba

    def set_objective(
            self, X_train, y_train, X_val, y_val,
            categorical_indicator
    ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.cat_ind = categorical_indicator

        self.model = self.get_model()  # Includes preprocessor

    def run(self, n_iter):
        # This is the function that is called to fit the model.
        # The param n_iter is defined if you change the sample strategy to
        # other value than "run_once"
        # https://benchopt.github.io/performance_curves.html
        self.model.fit(self.X_train, self.y_train)

    def get_result(self):
        # Returns the model after fitting.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(model=self.model)
