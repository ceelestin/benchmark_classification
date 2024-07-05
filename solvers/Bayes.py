from benchopt import safe_import_context

from benchmark_utils.bayes_estimator import BayesEstimator

with safe_import_context() as import_ctx:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder as OHE


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BayesEstimator):

    # Name to select the solver in the CLI and to display the results.
    name = 'Bayes'
    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'noise': [0, 10**-3, 10**-2, 10**-1, 1, 10, 100],
    }

    # Force solver to run only once if you don't want to record training steps
    sampling_strategy = "run_once"

    def get_model(self):
        size = self.X_train.shape[1]
        preprocessor = ColumnTransformer(
            [
                ("one_hot", OHE(
                        categories="auto", handle_unknown="ignore",
                    ), [i for i in range(size) if self.cat_ind[i]]),
                ("numerical", "passthrough",
                 [i for i in range(size) if not self.cat_ind[i]],)
            ]
        )

        return Pipeline(steps=[("preprocessor", preprocessor),
                               ("model", BayesEstimator(self.noise))])
