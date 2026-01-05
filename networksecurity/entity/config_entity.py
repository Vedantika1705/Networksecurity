from datetime import datetime
import os
from networksecurity.constants import traning_pipeline

class TraningPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = traning_pipeline.PIPELINE_NAME
        self.artifact_name = traning_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.model_dir = os.path.join("final_model")
        self.timestamp: str = timestamp


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TraningPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            traning_pipeline.DATA_INGESTION_DIR_NAME
        )

        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            traning_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            traning_pipeline.FILE_NAME
        )

        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            traning_pipeline.DATA_INGESTION_INGESTED_DIR,
            traning_pipeline.TRAIN_FILE_NAME
        )

        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            traning_pipeline.DATA_INGESTION_INGESTED_DIR,
            traning_pipeline.TEST_FILE_NAME
        )

        self.train_test_split_ratio: float = (
            traning_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        )

        self.collection_name: str = traning_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = traning_pipeline.DATA_INGESTION_DATABASE_NAME
