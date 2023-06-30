import os
import sys
from hr_analytics.logger import logging
from hr_analytics.exception import CustomException
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# from hr_analytics.components.data_transformation import DataTransformation, DataTransformationConfig
# from hr_analytics.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    logging.info("Created data ingestion config")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        
        try:
            df = pd.read_csv("E:\\iNeuron\\Projects\\HR_Analytics\\notebook\\datasets\\train_data.csv")
            logging.info("read the raw dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train Test split is now initiated")
            
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=111)
            
            logging.info("Train Test split is now completed")
            logging.info("Saving train set and test set as csv files")
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data Ingestion is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
                    
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    # data_transformation = DataTransformation()
    # train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    # model_trainer = ModelTrainer()
    # print(model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr))