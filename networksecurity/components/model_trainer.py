import os
import sys

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig



from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow
from urllib.parse import urlparse

#import dagshub
#dagshub.init(repo_owner='krishnaik06', repo_name='networksecurity', mlflow=True)

#os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/krishnaik06/networksecurity.mlflow"
#os.environ["MLFLOW_TRACKING_USERNAME"]="krishnaik06"
#os.environ["MLFLOW_TRACKING_PASSWORD"]="7104284f1bb44ece21e0e2adb4e36a250ae3251f"



class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def track_mlflow(self, best_model, classificationmetric):
        # MLflow disabled to avoid DAGsHub free plan experiment limit
            return


        
    def train_model(self, X_train, y_train, x_test, y_test):

        models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "AdaBoost": AdaBoostClassifier(),
        }

        params = {
            "Decision Tree": {
                "criterion": ["gini", "entropy"]
            },
            "Random Forest": {
                "n_estimators": [50, 100]
            },
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.01],
                "n_estimators": [50, 100]
            },
            "Logistic Regression": {},
            "AdaBoost": {
                "learning_rate": [0.1, 0.01],
                "n_estimators": [50, 100]
            }
        }

        # 🔹 Evaluate models (GridSearch happens here)
        model_report = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            models=models,
            param=params
        )

        # 🔹 Get best model name
        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]

        # 🔹 FIT THE BEST MODEL (IMPORTANT)
        best_model.fit(X_train, y_train)

        # 🔹 Train metrics
        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(
            y_true=y_train,
            y_pred=y_train_pred
        )

        # 🔹 Test metrics
        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(
            y_true=y_test,
            y_pred=y_test_pred
        )

        # 🔹 Track MLflow ONCE
        self.track_mlflow(best_model, classification_test_metric)

        # 🔹 Load preprocessor
        preprocessor = load_object(
            self.data_transformation_artifact.transformed_object_file_path
        )

        # 🔹 Create final_model folder
        os.makedirs("final_model", exist_ok=True)

        # 🔹 Create combined model
        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=best_model
        )

        # 🔹 Save combined model (THIS IS WHAT FASTAPI USES)
        save_object("final_model/model.pkl", network_model)

        # 🔹 Save preprocessor separately
        save_object("final_model/preprocessor.pkl", preprocessor)

        # 🔹 Trainer artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path="final_model/model.pkl",
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )

        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact



    
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise NetworkSecurityException(e,sys)