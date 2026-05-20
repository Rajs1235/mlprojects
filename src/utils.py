import os
import sys
from xml.parsers.expat import model
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}
        trained_models={}
        for i in range(len(models)):
            model=list(models.values())[i]
            param=params[list(models.keys())[i]]
            grid_search=GridSearchCV(estimator=model,param_grid=param,cv=3,n_jobs=-1,verbose=2)
            grid_search.fit(X_train,y_train)
            best_model=grid_search.best_estimator_

            y_train_pred=best_model.predict(X_train)
            y_test_pred=best_model.predict(X_test)
            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]]=(train_model_score,test_model_score)
            trained_models[list(models.keys())[i]]=best_model

            Randomsearch=RandomizedSearchCV(estimator=model,param_distributions=param,cv=3,n_jobs=-1,verbose=2)
            Randomsearch.fit(X_train,y_train)
            best_random_model=Randomsearch.best_estimator_
            y_train_pred_random=best_random_model.predict(X_train)
            y_test_pred_random=best_random_model.predict(X_test)
            train_random_model_score=r2_score(y_train,y_train_pred_random)
            test_random_model_score=r2_score(y_test,y_test_pred_random)
            report[list(models.keys())[i]+"_random"]=(train_random_model_score,test_random_model_score)
            trained_models[list(models.keys())[i]+"_random"]=best_random_model
            
        return report, trained_models
    except Exception as e:
        raise CustomException(e,sys)
def load_object(file_path): 
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)


    except Exception as e:
        raise CustomException(e,sys)