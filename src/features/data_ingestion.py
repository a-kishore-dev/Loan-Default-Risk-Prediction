import os
import polars as pl
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

class DataIngestionConfig(BaseModel):
    train_data_path: str = os.path.join("artifact", "train.parquet")
    test_data_path: str = os.path.join("artifact", "test.parquet")
    valid_data_path: str = os.path.join("artifact", "valid.parquet")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self, datapath):
        '''
        Read the data and combine multiple tables

        Parameters:
        datapath (str): The path where the dataset is located

        The train.parquet, valid.parquet and test.parquet saved to artifact for further processing.
        '''

        # Reading the data
        train_df = pl.read_csv(os.path.join(datapath, "application_train.csv"))
        test_df = pl.read_csv(os.path.join(datapath, "application_test.csv"))

        bureau = pl.read_csv(os.path.join(datapath, "bureau.csv"))
        bureau_balance = pl.read_csv(os.path.join(datapath, "bureau_balance.csv"))

        previous_application = pl.read_csv(os.path.join(datapath, "previous_application.csv"))

        # Extracting the features from the other table and combining to main table
        bureau_balance_features = (
            bureau_balance
                .group_by("SK_ID_BUREAU")
                .agg(
                    pl.len().alias("BUREAU_BALANCE_COUNT"),

                    pl.col("MONTHS_BALANCE")
                        .sum()
                        .alias("TOTAL_MONTHS_BALANCE"),

                    pl.col("STATUS")
                        .mode()
                        .first()
                        .alias("STATUS_FREQUENCY"),
                )
        )
        
        bureau = bureau.join(bureau_balance_features, on="SK_ID_BUREAU", how="left")

        bureau_features = (
            bureau
            .group_by("SK_ID_CURR")
            .agg(
                pl.len().alias("BUREAU_COUNT"),

                pl.col("AMT_CREDIT_SUM")
                    .mean()
                    .alias("AMT_CREDIT_MEAN"),
                
                (pl.col("CREDIT_ACTIVE") == "Active")
                    .sum()
                    .alias("ACTIVE_LOANS"),

                (pl.col("CREDIT_ACTIVE") != "Active")
                    .sum()
                    .alias("CLOSED_LOANS"),
                
                pl.col("BUREAU_BALANCE_COUNT")
                    .sum()
                    .alias("BUREAU_BALANCE_COUNT"),

                pl.col("TOTAL_MONTHS_BALANCE")
                    .sum()
                    .alias("TOTAL_MONTHS_BALANCE"),
                    
                pl.col("STATUS_FREQUENCY")
                    .mode()
                    .first()
                    .alias("STATUS_FREQUENCY"),
            )
        )

        previous_application_features = (
            previous_application
                .group_by("SK_ID_CURR")
                .agg(
                    pl.len().alias("PREVIOUS_APPLICATION_COUNT"),

                    pl.col("AMT_ANNUITY")
                        .sum()
                        .alias("TOTAL_AMT_ANNUITY"),
                    
                    pl.col("AMT_APPLICATION")
                        .sum()
                        .alias("TOTAL_AMT_APPLICATION"),
                    
                    pl.col("AMT_CREDIT")
                        .sum()
                        .alias("TOTAL_AMT_CREDIT"),
                    
                    pl.col("AMT_GOODS_PRICE")
                        .sum()
                        .alias("TOTAL_AMT_GOODS_PRICE"),
                    
                    (pl.col("FLAG_LAST_APPL_PER_CONTRACT") == "Y")
                        .sum()
                        .alias("APPROVED_LAST_APPL_PER_CONTRACT"),
                    
                    (pl.col("FLAG_LAST_APPL_PER_CONTRACT") == "N")
                        .sum()
                        .alias("NOT_APPROVED_LAST_APPL_PER_CONTRACT"),
                    
                    pl.col('NFLAG_LAST_APPL_IN_DAY')
                        .mode()
                        .first()
                        .alias("NFLAG_LAST_APPL_IN_DAY"),
                    
                    (pl.col("NAME_CONTRACT_STATUS") == "Approved")
                        .sum()
                        .alias("NAME_CONTRACT_STATUS_APPROVED"),
                    
                    (pl.col("NAME_CONTRACT_STATUS") == "Refused")
                        .sum()
                        .alias("NAME_CONTRACT_STATUS_REFUSED"),
                    
                    (pl.col("NAME_CONTRACT_STATUS") == "Canceled")
                        .sum()
                        .alias("NAME_CONTRACT_STATUS_CANCELED"),
                    
                    (pl.col("NAME_CONTRACT_STATUS") == "Unused offer")
                        .sum()
                        .alias("NAME_CONTRACT_STATUS_UNUSED"),
                    
                    pl.col('NFLAG_INSURED_ON_APPROVAL')
                        .mode()
                        .first()
                        .alias("NFLAG_INSURED_ON_APPROVAL"),
                )
        )

        train_df = train_df.join(bureau_features, on="SK_ID_CURR", how="left")
        test_df = test_df.join(bureau_features, on="SK_ID_CURR", how="left")

        train_df = train_df.join(previous_application_features, on="SK_ID_CURR", how="left", suffix="_PREVIOUS")
        test_df = test_df.join(previous_application_features, on="SK_ID_CURR", how="left", suffix="_PREVIOUS")

        train_df, valid_df = train_test_split(train_df, test_size=0.20, random_state=42)

        # Create the directory
        os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.data_ingestion_config.valid_data_path), exist_ok=True)

        # Saving the Data
        train_df.write_parquet(self.data_ingestion_config.train_data_path)
        test_df.write_parquet(self.data_ingestion_config.test_data_path)
        valid_df.write_parquet(self.data_ingestion_config.valid_data_path)
