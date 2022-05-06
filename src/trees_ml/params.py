from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import click

def find_best_params(features_train, features_val):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GradientBoostingClassifier())
    ])

    params_gbc ={
        'classifier__n_estimators': np.arange(50, 150,10),
        'classifier__max_depth': np.arange(1, 7),
        'classifier__learning_rate': [1,0.1,0.01],
        'classifier__random_state': np.arange(35, 45)
    }

    dtree_gscv = GridSearchCV(pipeline, params_gbc)
    dtree_gscv.fit(features_train, features_val)
    click.echo(f'GridSearchCV лучшие параметры:{dtree_gscv.best_params_}')
    return dtree_gscv.best_params_['classifier__n_estimators'],\
           dtree_gscv.best_params_['classifier__learning_rate'],\
           dtree_gscv.best_params_['classifier__max_depth'],\
           dtree_gscv.best_params_['classifier__random_state']