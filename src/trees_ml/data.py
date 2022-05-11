from pathlib import Path
from typing import Tuple

import click
import pandas as pd


def get_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    return features,target

def split_data(features,target,train_ix, test_ix):
    features_train, features_val = features.loc[train_ix,], features.loc[test_ix,]
    target_train, target_val = target[train_ix],target[test_ix]
    return features_train, features_val, target_train, target_val