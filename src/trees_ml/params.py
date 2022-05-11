from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import click
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss


def find_best_params(models_array, features_train, target_train,features_val,target_val, cv_inner):
    click.echo(f'Начинается подбор параметров:')
    outer_results ={}
    log_loss_result ={}
    for model in models_array:
        if(type(model["classifier"]).__name__ == "GradientBoostingClassifier"):
            params = {
                'classifier__n_estimators': np.arange(400, 600,100),
                'classifier__max_depth': np.arange(9, 11),
                'classifier__learning_rate': [0.1],
                'classifier__random_state': [42]
            }
        if(type(model["classifier"]).__name__ == "RandomForestClassifier"):
            params = {
                'classifier__n_estimators': np.arange(400, 600,100),
                'classifier__max_features': [2, 4, 6],
                'classifier__max_depth': np.arange(9, 11),
                'classifier__random_state': [42]
            }
        search = GridSearchCV(model, params, scoring='accuracy', cv=cv_inner, refit=True)
        result = search.fit(features_train, target_train)
        model1 = result.best_estimator_
        yhat = model1.predict(features_val)
        acc = accuracy_score(target_val, yhat)
        outer_results[type(model["classifier"]).__name__] = acc
    print(outer_results)
    return outer_results
    
