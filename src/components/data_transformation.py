import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data transformation initialized")

            categorical_cols=["cut","color","clarity"]
            numerical_cols=["carat","depth","table","x","y","z"]

            # Custom Ranking for ordinal variable
            cut_map= ["Fair", "Good", "Very Good", "Premium", "Ideal"]
            color_map = ["D", "E", "F", "G", "H", "I", "J"]
            clarity_map = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

            logging.info("Pipeline Initiated")
            # NUmerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())   
                ]
            )

            # Categorical Pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder',OrdinalEncoder(categories=[cut_map,color_map,clarity_map])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            logging.info("Pipeline Completed")
            return preprocessor

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read and test data comleted")
            logging.info(f"Train Dataframe Head: \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head: \n{test_df.head().to_string()}")
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformation_object()

            target_column_name="price"
            drop_columns=[target_column_name,'id']

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            # Transforming using preprocessor object
            logging.info("Applying preprocessing object on training datasets")
            input_feature_train_array=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_array,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_array,np.array(target_feature_test_df)]
            logging.info("Applying preprocessing object on training and testing datasets")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Pipeline pkl saved")
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_data_transformation")
