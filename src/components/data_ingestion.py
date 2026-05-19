import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
@dataclass
class DataIngestion:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            data_path = os.path.join(project_root, 'data', 'StudentsPerformance.csv')
            df=pd.read_csv(data_path)
            logging.info("Dataset read as pandas dataframe")

            os.makedirs(os.path.dirname(self.train_data_path),exist_ok=True)

            df.to_csv(self.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.train_data_path,index=False,header=True)
            test_set.to_csv(self.test_data_path,index=False,header=True)

            logging.info("Data Ingestion completed")

            return(
                self.train_data_path,
                self.test_data_path
            )
        except Exception as e:
            logging.info("Error occurred in Data Ingestion")
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    datatransformation=DataTransformation()
    datatransformation.initiate_data_transformation(train_data,test_data)