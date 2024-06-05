from benchopt import safe_import_context

#TODO other imports
with safe_import_context() as import_ctx:
    import optuna  # noqa: F401
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder as OHE
    from sklearn.preprocessing import OrdinalEncoder as OE
    from benchmark_utils.optuna_solver import OSolver
    from benchmark_utils.resnet_models import create_resnet_classifier_skorch


class Solver(OSolver):

    name = 'ResNet'
    requirements = ["pip:optuna"] #TODO

    extra_model_params = {}

    default_params = {
        "lr_scheduler": False,
        "module__activation": "reglu",
        "module__normalization": "batchnorm",
        "module__n_layers": 8,
        "module__d": 256,
        "module__d_hidden_factor": 2,
        "module__hidden_dropout": 0.2,
        "module__residual_dropout": 0.2,
        "lr": 1e-3,
        "optimizer__weight_decay": 1e-7,
        "optimizer": "adamw",
        "module__d_embedding": 128,
        "batch_size": 256,
        "max_epochs": 300,
        "use_checkpoints": True,
        "es_patience": 40,
        "lr_patience": 30,
        "verbose": 0,
    }

    def get_model(self):
        size = self.X_train.shape[1]
        preprocessor = ColumnTransformer(
            [
                ("one_hot", OE(
                        categories="auto", handle_unknown="use_encoded_value",
                        unknown_value=-1, #TODO: check
                    ), [i for i in range(size) if self.cat_ind[i]]),
                ("numerical", "passthrough", #TODO
                 [i for i in range(size) if not self.cat_ind[i]],)
            ]
        )
        pipe = Pipeline(steps= [("preprocessor", preprocessor),
                                  ("model", create_resnet_classifier_skorch(**self.default_params, n_classes=self.n_classes))])

        return pipe



    def sample_parameters(self, trial):
        n_layers = trial.suggest_int("module__n_layers", 1, 4, step=10)
        return dict(
            module__n_layers=n_layers
        )
