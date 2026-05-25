from airflow import DAG
from airflow.sdk import task
from datetime import datetime
import json
import sys
sys.path.insert(0, "/usr/local/airflow")  # Astro's working dir inside Docker

with DAG(
    dag_id="Credit_dag",
    start_date=datetime(2025, 6, 1),
    schedule="@weekly",
    catchup=False
):
    @task
    def data_ingestion_process():
        import sys
        sys.path.insert(0, "/usr/local/airflow")
        from src.features.data_ingestion import DataIngestion
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion(datapath="dataset")

    @task
    def data_transformation_process():
        import polars as pl
        import numpy as np
        import sys
        sys.path.insert(0, "/usr/local/airflow")
        from src.features.data_transformation import DataTransformation
        data_transformation = DataTransformation()
        train_data = pl.read_parquet("artifact/train.parquet")
        test_data = pl.read_parquet("artifact/test.parquet")
        valid_data = pl.read_parquet("artifact/valid.parquet")
        train_arr, valid_arr, test_arr, feature_names = data_transformation.initiate_data_transformation(
            train_data, valid_data, test_data
        )
        np.save("artifact/train_arr.npy", train_arr)
        np.save("artifact/valid_arr.npy", valid_arr)
        np.save("artifact/test_arr.npy", test_arr)

        with open("artifact/feature_names.json", "w") as f:
            json.dump(feature_names, f)

    @task
    def model_training_process():
        import polars as pl
        import numpy as np
        import sys
        sys.path.insert(0, "/usr/local/airflow")
        from src.features.model_training import ModelTrainer
        train = np.load("artifact/train_arr.npy")
        valid = np.load("artifact/valid_arr.npy")
        test = np.load("artifact/test_arr.npy")
        with open("artifact/feature_names.json", "r") as f:
            feature_names = json.load(f)
        model_trainer = ModelTrainer()
        best_model = model_trainer.initiate_model_trainer(train, valid, feature_names)
        test_pred = best_model.predict(test)
        result = pl.DataFrame({
            "target": test_pred.tolist()
        })
        result.write_parquet("artifact/predictions.parquet")

    ingestion = data_ingestion_process()
    transformed = data_transformation_process()
    transformed.set_upstream(ingestion)
    model_training = model_training_process()
    model_training.set_upstream(transformed)