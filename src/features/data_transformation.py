import os
import polars as pl
import numpy as np
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from src.features.utils import save_object

class DataTransformationConfig(BaseModel):
    preprocessor_obj_path: str = os.path.join("artifact","preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.numerical_columns = []
        self.categorical_columns = []

    
    def remove_outlier(self, df):
        '''
        Removing Outliers in the training data by applying winsorization to numerical column
        '''
        for column in self.numerical_columns:
            bounds = df.select(
                pl.col(column).quantile(0.025).alias("lower_bound"),
                pl.col(column).quantile(0.975).alias("upper_bound")
            )

            lower_limit = bounds["lower_bound"][0]
            upper_limit = bounds["upper_bound"][0]

            df = df.with_columns(
                pl.col(column)
                .clip(lower_bound=lower_limit, upper_bound=upper_limit)
                .alias(column)
            )
        return df
    
    def identify_column_type(self, df):
        '''
        Identify the columns type
        '''
        self.numerical_columns.clear()
        self.categorical_columns.clear()
        for col in df.columns:
            if df[col].dtype == pl.String:
                    self.categorical_columns.append(col)
            else:
                self.numerical_columns.append(col)
    
    def get_preprocessor_object(self, df):
        '''
        Return a preprocessor object that transforms the columns
        '''
        
        numerical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scalar", RobustScaler())
            ]
        )

        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoding", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("num_transformer", numerical_pipeline, self.numerical_columns),
                ("categorical_transformer", categorical_pipeline,self.categorical_columns),
            ]
        )

        return preprocessor


    def initiate_data_transformation(self, train_data, valid_data, test_data):
        '''
        Transform the data and return train, valid and test array.
        '''

        # Drop columns with missing percentage > 65
        missing_df = (
            train_data.null_count()
                .transpose(
                    include_header= True,
                    header_name="Column",
                    column_names=["missing_count"]
                )
                .with_columns(
                    (pl.col("missing_count") / len(train_data) * 100)
                    .alias("missing_percentage")
                )
                .filter(pl.col("missing_percentage") > 65)
        )

        train_data = train_data.drop(missing_df["Column"].to_list())
        valid_data = valid_data.drop(missing_df["Column"].to_list())
        test_data = test_data.drop(missing_df["Column"].to_list())

        input_feature_train_data = train_data.drop("TARGET")
        target_feature_train_data = train_data["TARGET"]

        input_feature_valid_data = valid_data.drop("TARGET")
        target_feature_valid_data = valid_data["TARGET"]

        self.identify_column_type(input_feature_train_data)

        # Removing outliers in the training data
        input_feature_train_data = self.remove_outlier(input_feature_train_data)        

        preprocessor_object = self.get_preprocessor_object(input_feature_train_data)

        input_feature_train_arr = preprocessor_object.fit_transform(input_feature_train_data.to_pandas())
        input_feature_valid_arr = preprocessor_object.transform(input_feature_valid_data.to_pandas())
        test_arr = preprocessor_object.transform(test_data.to_pandas())

        train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_data)]
        valid_arr = np.c_[input_feature_valid_arr, np.array(target_feature_valid_data)]

        feature_names = preprocessor_object.get_feature_names_out().tolist()
        feature_names = [name.split("__")[-1] for name in feature_names]

        save_object(
            preprocessor_object,
            self.data_transformation_config.preprocessor_obj_path
        )

        return (
            train_arr,
            valid_arr,
            test_arr,
            feature_names
        )
