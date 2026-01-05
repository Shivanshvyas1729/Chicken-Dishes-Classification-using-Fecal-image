
from cnnClassifier.config.configuration import ConfigurationManger
from cnnClassifier.components.data_ingestion import DataIngestion



class DataIngestionTrainingPipeline:
    def __init__(self):
        
        pass
    

    def main(self):
        
        config = ConfigurationManger()
        data_ingestion_config = config.get_data_ingestion_config()

        data_ingestion = DataIngestion(data_ingestion_config) 

        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
