import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor                      )
from sklearn.linear_model import LinearRegression   
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_obj_file_path):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "KNN":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False)
            }
            params = {

    "Linear Regression": {},

    "Decision Tree": {
        "max_depth": [5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },

    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt"]
    },

    "Gradient Boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0]
    },

    "KNN": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    },

    "XGBRegressor": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [1, 5]
    },

    "CatBoosting Regressor": {
        "iterations": [100, 200, 300],
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5]
    }
              }
          
        
            logging.info("Model training initiated")
            model_report, trained_models=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            best_model_name=max(model_report, key=lambda model_name: model_report[model_name][1])
            best_model_score=model_report[best_model_name][1]
            best_model=trained_models[best_model_name]
            logging.info(f"Best model found on both training and testing dataset: {best_model_name} with r2 score: {best_model_score}")
            if best_model_score < 0.6:
                logging.info("No best model found")
                raise CustomException("No best model found",sys)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            prediction=best_model.predict(X_test)
            r2_square=r2_score(y_test,prediction)
            return r2_square

        except Exception as e:
            logging.info("Error occurred in Model Training")
            raise CustomException(e,sys)
