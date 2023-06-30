import sys
import os

import numpy as np
import pandas as pd

from hr_analytics.exception import CustomException
from hr_analytics.logger import logging
from hr_analytics.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            return preds            
            
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 department:str,
                 region:str,
                 education:str,
                 gender:str,
                 recruitment_channel:str,
                 no_of_trainings:int,
                 age:int,
                 previous_year_rating:int,
                 length_of_service:int,
                 KPIs_met:str,
                 awards_won:str,
                 avg_training_score:int):
        
        self.department = department
        self.region = region
        self.education = education
        self.gender = gender
        self.recruitment_channel = recruitment_channel
        self.no_of_trainings = no_of_trainings
        self.age = age
        self.previous_year_rating = previous_year_rating
        self.length_of_service = length_of_service
        self.KPIs_met = KPIs_met
        self.awards_won = awards_won
        self.avg_training_score = avg_training_score
        
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "department":[self.department],
                "region":[self.region],
                "education":[self.education],
                "gender":[self.gender],
                "recruitment_channel":[self.recruitment_channel],
                "no_of_trainings":[self.no_of_trainings],
                "age":[self.age],
                "previous_year_rating":[self.previous_year_rating],
                "length_of_service":[self.length_of_service],
                "KPIs_met":[self.KPIs_met],
                "awards_won":[self.awards_won],
                "avg_training_score":[self.avg_training_score]                
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)