

from pathlib import Path
import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig



class Training:
    # PURPOSE:
    # This class handles the training stage of the ML pipeline.
    # It loads a prepared model, creates data generators,
    # trains the model, and saves the trained model.

    def __init__(self, config: TrainingConfig):
        # Store configuration so paths and hyperparameters
        # are not hardcoded inside the class
        self.config = config

    
    def get_base_model(self):
        # MEANING:
        # Load the updated base model created in the previous pipeline stage
        #
        # WHY:
        # Training should start from a prepared (transfer-learned) model,
        # not from scratch, to save time and improve accuracy
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )


    def train_valid_generator(self):
        # MEANING:
        # Create generators that load images from disk in batches
        #
        # WHY:
        # Using generators is memory-efficient and scalable
        # for large image datasets

        # Common preprocessing settings
        datagenerator_kwargs = dict(
            rescale=1. / 255,        # MEANING: Normalize pixel values to [0, 1]
                                     # WHY: Helps model train faster and more stably
            validation_split=0.20    # MEANING: Split data into train (80%) and val (20%)
                                     # WHY: Needed to evaluate model performance
        )

        # Image loading settings
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # MEANING: Resize images
                                                             # WHY: Model expects fixed input size
            batch_size=self.config.params_batch_size,        # MEANING: Images per batch
                                                             # WHY: Controls memory usage and speed
            interpolation="bilinear"                          # WHY: Good default for resizing images
        )

        # ===============================
        # VALIDATION DATA GENERATOR
        # ===============================

        # MEANING:
        # Validation data generator is used ONLY for evaluation
        #
        # WHY:
        # Validation data must represent real, untouched data.
        # Augmentation would distort images and give misleading accuracy.

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",     # MEANING: Use validation split
            shuffle=False,           # WHY: Keep order fixed for consistent evaluation
            **dataflow_kwargs
        )

        # ===============================
        # TRAINING DATA GENERATOR
        # ===============================

        # MEANING:
        # Training data generator is used for learning
        #
        # WHY:
        # Augmentation helps the model generalize better
        # and reduces overfitting

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,      # WHY: Make model rotation-invariant
                horizontal_flip=True,   # WHY: Handle left-right variations
                width_shift_range=0.2,  # WHY: Improve robustness to translations
                height_shift_range=0.2,
                shear_range=0.2,        # WHY: Handle geometric distortions
                zoom_range=0.2,         # WHY: Improve scale invariance
                **datagenerator_kwargs
            )
        else:
            # MEANING:
            # If augmentation is disabled, reuse validation generator settings
            #
            # WHY:
            # Keeps preprocessing consistent while avoiding extra transformations
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",       # MEANING: Use training split
            shuffle=True,            # WHY: Shuffle improves learning
            **dataflow_kwargs
        )

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        # MEANING:
        # Save model to disk
        #
        # WHY:
        # Trained models are artifacts needed for evaluation,
        # inference, and deployment
        model.save(path)


    def train(self):
        # MEANING:
        # Train the model using training and validation generators
        #
        # WHY:
        # Generator-based training is memory-efficient
        # and suitable for large datasets

        # Calculate number of steps per epoch
        self.steps_per_epoch = (
            self.train_generator.samples // self.train_generator.batch_size
        )

        self.validation_steps = (
            self.valid_generator.samples // self.valid_generator.batch_size
        )

        # Train the model
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,      # WHY: Controls training duration
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator   # WHY: Monitor generalization
        )

        # Save the final trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
