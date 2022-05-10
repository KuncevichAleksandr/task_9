from pathlib import Path
import click
from joblib import dump
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

from .data import get_dataset
from .data import split_data
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from .pipeline import create_pipeline
from .params import find_best_params
from .ModelFactory import ModelFactory

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--n-estimators",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--learning-rate",
    default=0.1,
    type=float,
    show_default=True,
)
@click.option(
    "--max-depth",
    default=3,
    type=int,
    show_default=True,
)
@click.option(
    "--use-grid-search-cv",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--use-gradient-boosting-classifier",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--use-random-forest-classifier",
    default=False,
    type=bool,
    show_default=True,
)

def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    use_grid_search_cv:bool,
    use_gradient_boosting_classifier:bool,
    use_random_forest_classifier:bool
    ) -> None:
    features,target = get_dataset(dataset_path)
    models_array = []
    with mlflow.start_run():
        if(use_gradient_boosting_classifier):
            models_array.append(ModelFactory.buid("GradientBoostingClassifier"))
        if(use_random_forest_classifier):
            models_array.append(ModelFactory.buid("RandomForestClassifier"))

        cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
        outer_results = list()
        for train_ix, test_ix in cv_outer.split(features):
            features_train, features_val, target_train, target_val = split_data(features,target,train_ix, test_ix)
            cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
            # if use_grid_search_cv:

            for model in models_array:
                result = model.fit(features_train, target_train)
                print(result)
        #         best_model = result.best_estimator_
                yhat = result.predict(features_val)
                acc = accuracy_score(target_val, yhat)
                outer_results.append(acc)
        print(outer_results)

        # if use_grid_search_cv:
        #     n_estimators, learning_rate, max_depth, random_state = find_best_params(features_train, target_train)
        # pipeline = create_pipeline(use_scaler, n_estimators, learning_rate, max_depth, random_state)
        # pipeline.fit(features_train, target_train)
        # predict_vals = pipeline.predict_proba(features_val)
        # log_loss_val = log_loss(target_val,predict_vals)
        # accuracy = accuracy_score(target_val, pipeline.predict(features_val))
        # mlflow.log_param("use_scaler", use_scaler)
        # mlflow.log_param("use_grid_search_cv", use_grid_search_cv)
        # mlflow.log_param("n_estimators", n_estimators)
        # mlflow.log_param("learning_rate", learning_rate)
        # mlflow.log_param("max_depth", max_depth)
        # mlflow.log_metric("accuracy", accuracy)
        # mlflow.log_metric("log_loss", log_loss_val)
        # click.echo(f"Accuracy: {accuracy}.")
        # dump(pipeline, save_model_path)
        # click.echo(f"Model is saved to {save_model_path}.")