
import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path


class PrepareBaseModel:
    """
    This class is responsible for:
    1. Loading a pretrained base model (VGG16)
    2. Freezing layers if needed
    3. Adding custom layers on top
    4. Saving both base and updated models
    """

    def __init__(self, config: PrepareBaseModelConfig):
        # Store configuration object (paths, parameters, hyperparameters)
        self.config = config


    def get_base_model(self):
        """
        Loads the VGG16 pretrained model using parameters from config
        and saves it to disk.
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,   # Input image size
            weights=self.config.params_weights,          # Pretrained weights (e.g., 'imagenet')
            include_top=self.config.params_include_top   # Whether to include original classifier
        )

        # Save the base model to the specified path
        self.save_model(path=self.config.base_model_path, model=self.model)


    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepares the full model by:
        - Freezing layers of the base model
        - Adding custom classification layers
        - Compiling the model
        """

        # Freeze all layers of the base model if freeze_all is True
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False

        # Freeze layers up to a certain point if freeze_till is provided
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Flatten the output of the base model
        flatten_in = tf.keras.layers.Flatten()(model.output)

        # Add final dense layer for classification
        prediction = tf.keras.layers.Dense(
            units=classes,              # Number of output classes
            activation="softmax"        # Softmax for multi-class classification
        )(flatten_in)

        # Create the full model
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Compile the model
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        # Print model architecture
        full_model.summary()

        return full_model
    

    def update_base_model(self):
        """
        Creates the final model by adding custom layers on top of the base model
        and saves the updated model.
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,                  # Freeze all base model layers
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        # Save the updated model
        self.save_model(
            path=self.config.updated_base_model_path,
            model=self.full_model
        )


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the model to the given path.
        """
        model.save(path)
