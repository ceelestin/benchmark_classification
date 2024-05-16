from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import openml
    from sklearn.model_selection import train_test_split


DATASETS = {
    'covertype': 44121,
    #'covertype_categorical': 44159,
}


class Dataset(BaseDataset):

    name = 'covertype'

    install_cmd = 'conda'
    requirements = ["pip:openml", "pip:chardet"]

    parameters = {
        "dataset": list(DATASETS),
    }

    def get_data(self):
        dataset = openml.datasets.get_dataset(
            self.dataset
        )
        X, y, cat_indicator, attribute_names = dataset.get_data(
            dataset_format="array", target=dataset.default_target_attribute
        )

        # Downsample the data while maintaining the proportion of each label
        X, _, y, _ = train_test_split(X, y, test_size=580012, stratify=y, random_state=42)


        return dict(
            X=X,
            y=y,
            categorical_indicator=cat_indicator
        )