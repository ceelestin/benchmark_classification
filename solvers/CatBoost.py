from benchopt import safe_import_context
from benchmark_utils.optuna_solver import OSolver

with safe_import_context() as import_ctx:
    import optuna  # noqa: F401
    from catboost import CatBoostClassifier

class Solver(OSolver):

    name = 'CatBoost'

    requirements = ['pip:catboost']

    def get_model(self):
        #cat_features=[i for i in range(len(self.cat_ind)) if self.cat_ind[i]]
        #return CatBoostClassifier(cat_features=cat_features)
         return CatBoostClassifier()

    def sample_parameters(self, trial):
        iterations = trial.suggest_int("iterations", 10, 200, step=10)
        #learning_rate = trial.suggest_float("learning_rate", 1e-2, 1, log=True)
        depth = trial.suggest_int("depth", 3, 50)
        
        return dict(
            iterations=iterations,
            #learning_rate=learning_rate,
            depth=depth,
            
        )


