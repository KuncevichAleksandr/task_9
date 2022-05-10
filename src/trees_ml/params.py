from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import click

def find_best_params(features_train, target_train):
    click.echo(f'Начинается подбор параметров:')
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GradientBoostingClassifier())
    ])

    params_gbc ={
        'classifier__n_estimators': np.arange(400, 600,100),
        'classifier__max_depth': np.arange(9, 11),
        'classifier__learning_rate': [0.1],
        'classifier__random_state': [42]
    }

    dtree_gscv = GridSearchCV(pipeline, params_gbc)
    dtree_gscv.fit(features_train, target_train)
    click.echo(f'GridSearchCV лучшие параметры:{dtree_gscv.best_params_}')
    return dtree_gscv.best_params_['classifier__n_estimators'],\
           dtree_gscv.best_params_['classifier__learning_rate'],\
           dtree_gscv.best_params_['classifier__max_depth'],\
           dtree_gscv.best_params_['classifier__random_state']