import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tabulate import tabulate
import time
import os 
from pathlib import Path

file_path = r"output_images"

rootPath = os.getcwd() 
prevRootPath = str(Path(rootPath).parents[0])
# Directory with dataset
dataset_dir = prevRootPath + "\\" + file_path

# Hyperparameters
batch_size = 32
img_height = 30
img_width = 42
epochs = 10
trials = 10

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# Define different configurations
configurations = [
    {'conv_layers': [16, 32], 'dense_units': [128], 'dropout': 0.5, 'optimizer': 'adam', 'learning_rate': 2e-4},
    {'conv_layers': [32, 64], 'dense_units': [256], 'dropout': 0.3, 'optimizer': 'sgd', 'learning_rate': 1e-4},
    {'conv_layers': [64, 128], 'dense_units': [512], 'dropout': 0.4, 'optimizer': 'adam', 'learning_rate': 5e-4},
    {'conv_layers': [16, 32, 64], 'dense_units': [256], 'dropout': 0.5, 'optimizer': 'adam', 'learning_rate': 1e-4},
    {'conv_layers': [32, 64, 128], 'dense_units': [128], 'dropout': 0.3, 'optimizer': 'sgd', 'learning_rate': 2e-4},
    {'conv_layers': [16, 64, 128], 'dense_units': [128, 64], 'dropout': 0.5, 'optimizer': 'adam', 'learning_rate': 1e-3},
    {'conv_layers': [32, 64], 'dense_units': [256, 128], 'dropout': 0.2, 'optimizer': 'sgd', 'learning_rate': 5e-4},
    {'conv_layers': [64, 128, 256], 'dense_units': [512, 256], 'dropout': 0.4, 'optimizer': 'adam', 'learning_rate': 1e-4},
    {'conv_layers': [16, 32, 64, 128], 'dense_units': [256], 'dropout': 0.3, 'optimizer': 'sgd', 'learning_rate': 2e-4},
    {'conv_layers': [32, 64, 128, 256], 'dense_units': [128, 64], 'dropout': 0.5, 'optimizer': 'adam', 'learning_rate': 5e-5}
]


trialResults = []

for trial in range(trials):
    print(f"Starting Trial {trial}")

    analysis = [['Config', 'Trainable Params', 'Training Time', 'Accuracy', 'Validation Accuracy', 'Loss', 'Validation Loss']]

    for idx, config in enumerate(configurations):
        print(f"Testing Configuration {idx + 1}")

        model = Sequential([
            layers.Rescaling(1./255, input_shape=(img_height, img_width, 3))
        ])

        # Add convolutional layers
        for filters in config['conv_layers']:
            model.add(layers.Conv2D(filters, 3, padding='same', activation='relu'))
            model.add(layers.MaxPooling2D())

        model.add(layers.Flatten())

        # Add dense layers
        for units in config['dense_units']:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(config['dropout']))

        model.add(layers.Dense(num_classes))

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate']) if config['optimizer'] == 'adam' else keras.optimizers.SGD(learning_rate=config['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        stime = time.time()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=0
        )
        etime = time.time()

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        analysis.append([
            idx + 1,
            np.sum([tf.keras.backend.count_params(p) for p in model.trainable_variables]),
            round(etime-stime, 5),
            round(acc[-1], 3),
            round(val_acc[-1], 3),
            round(loss[-1], 3),
            round(val_loss[-1], 3)
        ])

        del model

    trialResults.append(analysis)

def average_tables(tables):
    numeric_tables = [np.array(table[1:], dtype=float) for table in tables]
    stacked_tables = np.stack(numeric_tables, axis=0)
    averaged_table = np.mean(stacked_tables, axis=0)
    header = tables[0][0]
    result = [header] + averaged_table.tolist()
    return result

def print_best_model_configuration(trial_results):
    # Flatten the trial results into a single list
    flattened_results = [result for trial in trial_results for result in trial[1:]]

    # Find the index of the best configuration based on validation accuracy
    best_result = max(flattened_results, key=lambda x: x[4])  # x[4] is the validation accuracy

    # Extract information
    best_config_idx = best_result[0] - 1  # Convert to zero-based index
    best_trainable_params = best_result[1]
    best_training_time = best_result[2]
    best_accuracy = best_result[3]
    best_val_accuracy = best_result[4]
    best_loss = best_result[5]
    best_val_loss = best_result[6]
    
    best_config = configurations[best_config_idx]

    # Print the results
    print("Best Model Configuration:")
    print(f"Configuration Index: {best_config_idx + 1}")
    print(f"Convolutional Layers: {best_config['conv_layers']}")
    print(f"Dense Units: {best_config['dense_units']}")
    print(f"Dropout Rate: {best_config['dropout']}")
    print(f"Optimizer: {best_config['optimizer']}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Trainable Parameters: {best_trainable_params}")
    print(f"Training Time: {best_training_time} seconds")
    print(f"Training Accuracy: {best_accuracy}")
    print(f"Validation Accuracy: {best_val_accuracy}")
    print(f"Training Loss: {best_loss}")
    print(f"Validation Loss: {best_val_loss}")

print(tabulate(average_tables(trialResults), headers='firstrow', tablefmt='fancy_grid'))
# Function to identify the best configuration
def find_best_configuration(average_results):
    header = average_results[0]
    data = average_results[1:]
    # Find index of the best configuration based on maximum validation accuracy
    best_index = np.argmax([row[4] for row in data])  # Index with highest validation accuracy
    return best_index, data[best_index]

# Rebuild the best model and print summary
def build_and_summarize_best_model(config):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3))
    ])
    
    # Add convolutional layers
    for filters in config['conv_layers']:
        model.add(layers.Conv2D(filters, 3, padding='same', activation='relu'))
        model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())

    # Add dense layers
    for units in config['dense_units']:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(config['dropout']))

    model.add(layers.Dense(num_classes))

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate']) if config['optimizer'] == 'adam' else tf.keras.optimizers.SGD(learning_rate=config['learning_rate'])
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    print(f"Best Configuration Summary:")
    model.summary()

# Main script
# Average results
print("Calculating average results...")
average_results = average_tables(trialResults)

# Identify best configuration
best_index, best_config = find_best_configuration(average_results)
print(f"\nBest Configuration (Index: {best_index + 1}):")
print(best_config)

# Rebuild and summarize the best model
print("\nBuilding and summarizing the best model...")
build_and_summarize_best_model(configurations[best_index])