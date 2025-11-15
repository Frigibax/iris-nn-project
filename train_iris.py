import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
import time
import os
import json
import logging
from datetime import datetime

# --- Configuration and Constants ---

# Define the base configuration for the project
BASE_CONFIG = {
    "DATASET_NAME": "Iris_Flower_Land_Data",
    "INPUT_FEATURES": 4,
    "OUTPUT_CLASSES": 3,
    "RANDOM_SEED": 42,
    "TEST_SPLIT": 0.2,
    "LOG_LEVEL": logging.INFO,
    # Add early stopping patience so callbacks work
    "EARLY_STOPPING_PATIENCE": 5
}

# Hyperparameter grid for the search function (adds complexity)
HYPERPARAMETER_GRID = {
    "hidden_neurons": [16, 24, 32],
    "epochs": [50, 100],
    "batch_size": [4, 8],
    "learning_rate": [0.001, 0.01]
}

# Define file paths
PROJECT_ROOT = "iris_nn_project"
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
SCALER_PARAMS_PATH = os.path.join(MODELS_DIR, "scaler_params.json")
BEST_MODEL_PATH_BASE = os.path.join(MODELS_DIR, "best_iris_nn.keras")
GRID_SEARCH_REPORT_PATH = os.path.join(LOGS_DIR, "grid_search_report.json")

# --- Setup Logging and Directories ---

def setup_project_environment():
    """Initializes directories and configures the Python logger."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Configure logging to a file and console
    log_filename = os.path.join(LOGS_DIR, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=BASE_CONFIG["LOG_LEVEL"],
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Project environment setup complete. Logs being saved to {log_filename}")
    np.random.seed(BASE_CONFIG["RANDOM_SEED"])
    tf.random.set_seed(BASE_CONFIG["RANDOM_SEED"])

# --- Helper Functions for Data Serialization ---

def save_scaler_params(scaler, path):
    """Saves the mean and variance of the StandardScaler for later use."""
    try:
        scaler_params = {
            "mean": np.asarray(scaler.mean_).tolist(),
            "scale": np.asarray(scaler.scale_).tolist(),
            "n_features_in": int(getattr(scaler, 'n_features_in_', scaler.mean_.shape[0]))
        }
        with open(path, 'w') as f:
            json.dump(scaler_params, f, indent=4)
        logging.info(f"Scaler parameters saved to {path}")
    except Exception as e:
        logging.error(f"Error saving scaler parameters: {e}")

# --- Data Preparation Class ---

class DataProcessor:
    """Handles all data loading, splitting, and scaling operations."""
    def __init__(self, config):
        self.config = config
        self.iris = load_iris()
        self.class_names = self.iris.target_names
        self.scaler = None
        
    def preprocess(self):
        """Executes the full preprocessing pipeline."""
        X, y = self.iris.data, self.iris.target
        
        logging.info(f"Data loaded: {len(X)} samples, {X.shape[1]} features.")

        # 1. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config["TEST_SPLIT"], 
            random_state=self.config["RANDOM_SEED"], 
            stratify=y
        )
        
        # 2. Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 3. Save scaler for frontend use
        save_scaler_params(self.scaler, SCALER_PARAMS_PATH)

        # 4. One-hot encode labels
        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=self.config["OUTPUT_CLASSES"])
        y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=self.config["OUTPUT_CLASSES"])
        
        logging.info(f"Training/Test split: {len(X_train_scaled)} / {len(X_test_scaled)} samples.")
        return X_train_scaled, X_test_scaled, y_train_onehot, y_test_onehot, y_test

# --- Neural Network Class (The Core Model) ---

class IrisClassifierNN:
    """An object-oriented wrapper for the Keras neural network model."""
    
    def __init__(self, input_shape, output_classes, hidden_neurons=16, learning_rate=0.001):
        self.hidden_neurons = hidden_neurons
        self.model = self._build_model(input_shape, output_classes, learning_rate)
        
    def _build_model(self, input_shape, output_classes, learning_rate):
        """Constructs and compiles the Keras model."""
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.hidden_neurons, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(int(self.hidden_neurons / 2), activation='relu'),
            tf.keras.layers.Dense(output_classes, activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary(print_fn=lambda x: logging.info(f"Model Summary Line: {x}"))
        total_params = model.count_params()
        logging.info(f"Total Model Parameters (Weights + Biases): {total_params}.")
        
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        """Trains the model and uses callbacks for best model saving."""
        
        model_save_path = os.path.join(MODELS_DIR, f"temp_best_model_{int(time.time())}.keras")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=BASE_CONFIG["EARLY_STOPPING_PATIENCE"], verbose=1, mode='min'),
            tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy', save_best_only=True, verbose=0, mode='max')
        ]
        
        history = self.model.fit(
            X_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        try:
            # If ModelCheckpoint saved a better model, load it
            if os.path.exists(model_save_path):
                self.model = tf.keras.models.load_model(model_save_path)
                logging.info(f"Loaded best model weights from {model_save_path}")
                try:
                    os.remove(model_save_path)
                except Exception:
                    pass
        except Exception as e:
            logging.warning(f"Could not load best model weights. Using final epoch weights. Error: {e}")

        return history.history

    def evaluate(self, X_test, y_test, y_true_labels, class_names):
        """Evaluates the model and prints detailed reports."""
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"Test Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%")
        
        y_pred_probs = self.model.predict(X_test)
        y_pred_labels = np.argmax(y_pred_probs, axis=1)
        
        report = classification_report(y_true_labels, y_pred_labels, target_names=class_names, output_dict=True)
        conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
        
        logging.info(f"\n--- Classification Report ---\n{classification_report(y_true_labels, y_pred_labels, target_names=class_names)}")
        logging.info(f"\n--- Confusion Matrix ---\n{conf_matrix}")
        
        return accuracy, report, conf_matrix

# --- Hyperparameter Search Function ---

def run_hyperparameter_search(X_train, X_test, y_train_onehot, y_test_onehot, y_test_labels, class_names, grid):
    logging.info("\n--- Starting Hyperparameter Grid Search ---")
    best_accuracy = 0
    best_params = {}
    search_results = []
    
    X_search_train, X_search_val, y_search_train, y_search_val = train_test_split(
        X_train, y_train_onehot, test_size=0.1, random_state=BASE_CONFIG["RANDOM_SEED"]
    )

    total_runs = (len(grid['hidden_neurons']) * len(grid['epochs']) * len(grid['batch_size']) * len(grid['learning_rate']))
    run_counter = 0
    
    for hn in grid['hidden_neurons']:
        for e in grid['epochs']:
            for bs in grid['batch_size']:
                for lr in grid['learning_rate']:
                    run_counter += 1
                    params = {'hidden_neurons': hn, 'epochs': e, 'batch_size': bs, 'learning_rate': lr}
                    logging.info(f"\nGrid Search Run {run_counter}/{total_runs}: {params}")

                    try:
                        nn_model = IrisClassifierNN(
                            input_shape=BASE_CONFIG["INPUT_FEATURES"], 
                            output_classes=BASE_CONFIG["OUTPUT_CLASSES"], 
                            hidden_neurons=hn, 
                            learning_rate=lr
                        )
                        nn_model.train(X_search_train, y_search_train, X_search_val, y_search_val, e, bs)
                        
                        accuracy, report, _ = nn_model.evaluate(
                            X_test, y_test_onehot, y_test_labels, class_names
                        )
                        
                        result = {
                            "params": params,
                            "test_accuracy": float(accuracy),
                            "report_macro_avg": report.get('macro avg', {})
                        }
                        search_results.append(result)

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = params
                            nn_model.model.save(BEST_MODEL_PATH_BASE)
                            logging.info(f"NEW BEST MODEL FOUND! Accuracy: {best_accuracy:.4f}")

                    except Exception as ex:
                        logging.error(f"Error during run with params {params}: {ex}")
                        logging.exception(ex)
                        
    try:
        with open(GRID_SEARCH_REPORT_PATH, 'w') as f:
            json.dump(search_results, f, indent=4)
    except Exception as e:
        logging.error(f"Could not write grid search report: {e}")
        
    logging.info(f"\n--- Grid Search Complete ---")
    logging.info(f"Best Accuracy: {best_accuracy:.4f}")
    logging.info(f"Best Parameters: {best_params}")
    logging.info(f"Search results saved to {GRID_SEARCH_REPORT_PATH}")
    
    return best_params

# --- Main Execution Block ---

def main_training_pipeline():
    """Orchestrates the entire machine learning pipeline."""
    
    setup_project_environment()
    start_time = time.time()
    
    logging.info(f"Project starting on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    data_processor = DataProcessor(BASE_CONFIG)
    X_train, X_test, y_train_onehot, y_test_onehot, y_test_labels = data_processor.preprocess()
    class_names = data_processor.class_names

    best_params = run_hyperparameter_search(
        X_train, X_test, y_train_onehot, y_test_onehot, y_test_labels, class_names, HYPERPARAMETER_GRID
    )
    
    if os.path.exists(BEST_MODEL_PATH_BASE):
        final_model = tf.keras.models.load_model(BEST_MODEL_PATH_BASE)
        logging.info(f"\nLoaded Best Model for final reporting from {BEST_MODEL_PATH_BASE}")
    else:
        logging.error("Could not find the best model file. Training pipeline failed.")
        return

    loss, accuracy = final_model.evaluate(X_test, y_test_onehot, verbose=0)
    logging.info(f"\n--- FINAL MODEL REPORT ---")
    logging.info(f"Optimal Parameters: {best_params}")
    logging.info(f"FINAL Test Accuracy: {accuracy*100:.2f}%")
    
    y_pred_probs = final_model.predict(X_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    logging.info(f"\n--- Final Classification Report ---\n{classification_report(y_test_labels, y_pred_labels, target_names=class_names)}")

    end_time = time.time()
    total_time = end_time - start_time
    
    logging.info("\n--------------------------------------------------")
    logging.info("Full Project Pipeline Complete!")
    logging.info(f"Total execution time: {total_time:.2f} seconds.")
    logging.info(f"Best Model saved at: {BEST_MODEL_PATH_BASE}")
    logging.info("Run the Streamlit frontend with 'streamlit run app.py' (if available)")

if __name__ == "__main__":
    main_training_pipeline()
