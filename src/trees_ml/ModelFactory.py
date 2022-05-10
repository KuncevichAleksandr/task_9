from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

class ModelFactory:
    def buid(name,params = {"random_state":1}):
        if(name == "GradientBoostingClassifier"):
            return GradientBoostingClassifier(params)
        if(name == "RandomForestClassifier"):
            return RandomForestClassifier(params)
        # self.models_array.append(GradientBoostingClassifier(random_state=42))
