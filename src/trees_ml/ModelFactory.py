from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from .pipeline import create_pipeline

class ModelFactory:
    def buid(name,use_scaler,params = {"random_state":1}):
        if(name == "GradientBoostingClassifier"):
            return create_pipeline(use_scaler,GradientBoostingClassifier(**params))
        if(name == "RandomForestClassifier"):
            return create_pipeline(use_scaler,RandomForestClassifier(**params))

