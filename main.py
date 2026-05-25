import polars as pl
from src.features.data_ingestion import DataIngestion
from src.features.data_transformation import DataTransformation
from src.features.model_training import ModelTrainer

def predict_pipeline():
    '''
    Create a pipeline to train the model and predict the data
    '''
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion(datapath="dataset")

    data_transformation = DataTransformation()
    train_data = pl.read_parquet("artifact/train.parquet")
    test_data = pl.read_parquet("artifact/test.parquet")
    valid_data = pl.read_parquet("artifact/valid.parquet")
    train_arr, valid_arr, test_arr, feature_names = data_transformation.initiate_data_transformation(
        train_data, valid_data, test_data
    )

    model_trainer = ModelTrainer()
    best_model = model_trainer.initiate_model_trainer(train_arr, valid_arr, feature_names)
    test_pred = best_model.predict(test_arr)
    result = pl.DataFrame({
        "target": test_pred.tolist()
    })
    result.write_parquet("artifact/predictions.parquet")

if __name__ == "__main__":
    predict_pipeline()