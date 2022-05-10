from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

class ModelFactory:
    def __init__(self, models_name_array):
        self.models_array = []
        for model_name in models_name_array:
            self.models_array.append(eval(model_name))
