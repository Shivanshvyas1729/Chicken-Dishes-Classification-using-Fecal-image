import tensorflow as tf
import mlflow
import mlflow.keras
from urllib.parse import urlparse 
from cnnClassifier.utils.common import save_json 
from cnnClassifier.entity.config_entity import EvaluationConfig
from pathlib import Path

import dagshub
dagshub.init(repo_owner='shivanshvyas29', repo_name='Chicken-Dishes-Classification-using-Fecal-image', mlflow=True)




class Evaluation:
    def __init__(self, config: EvaluationConfig):
        # Store the configuration object which contains
        # paths, parameters, and MLflow settings
        self.config = config

    
    def _valid_generator(self):
        """
        This method creates a validation data generator.
        It is separated into its own function to keep the code modular
        and reusable.
        """

        # ImageDataGenerator arguments
        # rescale: Normalizes pixel values from [0,255] to [0,1]
        # validation_split: Reserves 30% of training data for validation
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        # Data flow arguments
        # target_size: Resizes images to match model input size
        # batch_size: Number of images processed at once
        # interpolation: Method used for resizing images
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Create ImageDataGenerator object for validation
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Load images from directory for validation
        # shuffle=False is used to keep order consistent during evaluation
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Static method to load a trained Keras model from disk.
        Static method is used because this function does not
        depend on class attributes.
        """
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        """
        This method evaluates the trained model on validation data.
        """

        # Load the trained model from the given path
        self.model = self.load_model(self.config.path_of_model)

        # Prepare validation data generator
        self._valid_generator()

        # Evaluate the model on validation data
        # Returns loss and accuracy
        self.score = self.model.evaluate(self.valid_generator)

    
    def save_score(self):
        """
        Saves the evaluation scores (loss and accuracy)
        into a JSON file for future reference.
        """

        scores = {
            "loss": self.score[0],
            "accuracy": self.score[1]
        }

        # Save scores as a JSON file
        save_json(path=Path("scores.json"), data=scores)

    

    def log_into_mlflow(self):
        """
        Logs parameters, metrics, and model into MLflow
        for experiment tracking and versioning.
        """

        # Set MLflow tracking URI
        mlflow.set_registry_uri(self.config.mlflow_uri)

        # Get tracking store type (file or database)
        tracking_url_type_store = urlparse(
            mlflow.get_tracking_uri()  #This line checks whether MLflow 
                                        #is using a local file system or a remote tracking server.
        ).scheme
        
        # Start an MLflow run
        with mlflow.start_run():

            # Log all hyperparameters
            mlflow.log_params(self.config.all_params)

            # Log evaluation metrics
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )

            # Model Registry does not support file-based storage
            if tracking_url_type_store != "file":

                # Register the model in MLflow Model Registry
                mlflow.keras.log_model(
                    self.model,
                    "model",
                    registered_model_name="VGG16Model"
                )
            else:
                # Log model without registration
                mlflow.keras.log_model(self.model, "model")    
