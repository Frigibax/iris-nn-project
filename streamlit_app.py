import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import json
import os
import time

# --- Configuration and File Paths (Must match Part 1) ---
PROJECT_ROOT = "iris_nn_project"
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SCALER_PARAMS_PATH = os.path.join(MODELS_DIR, "scaler_params.json")
BEST_MODEL_PATH_BASE = os.path.join(MODELS_DIR, "best_iris_nn.keras")

# Hardcoded class names (as they are fixed for the Iris dataset)
CLASS_NAMES = np.array(['setosa', 'versicolor', 'virginica'])

# --- Resource Loading with Streamlit Caching ---

@st.cache_resource 
def load_model_and_scaler():
    """Loads the best model and reconstructs the StandardScaler object."""
    try:
        start_time = time.time()
        
        # 1. Load Keras Model
        model = tf.keras.models.load_model(BEST_MODEL_PATH_BASE)
        
        # 2. Load Scaler Parameters
        with open(SCALER_PARAMS_PATH, 'r') as f:
            scaler_params = json.load(f)
        
        # 3. Reconstruct StandardScaler
        scaler = StandardScaler()
        # Set the mean and scale (std dev) directly to the empty scaler
        scaler.mean_ = np.array(scaler_params['mean'])
        scaler.scale_ = np.array(scaler_params['scale'])
        scaler.n_features_in_ = scaler_params['n_features_in']
        
        st.sidebar.success(f"Model and Scaler loaded successfully in {time.time()-start_time:.2f}s!")
        return model, scaler

    except FileNotFoundError as e:
        st.error(f"Error: Required file not found. Have you run the training script yet? Missing file: {e.filename}")
        st.info("Please run the main Python training script (Part 1) first to generate the necessary 'models/' files.")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during resource loading: {e}")
        return None, None

# Load resources globally
MODEL, SCALER = load_model_and_scaler()

# --- Prediction Function ---

def make_prediction(model, scaler, input_data):
    """Scales input data and returns the prediction result."""
    # Convert list of inputs to a NumPy array and reshape for the model
    input_array = np.array([input_data], dtype=np.float32)
    
    # Apply the loaded scaler transformation
    scaled_data = scaler.transform(input_array)
    
    # Get probability predictions
    probabilities = model.predict(scaled_data, verbose=0)[0]
    
    # Find the predicted class index
    predicted_class_index = np.argmax(probabilities)
    
    return CLASS_NAMES[predicted_class_index], probabilities

# --- Streamlit App Layout ---

def main():
    st.set_page_config(page_title="Small NN Iris Classifier", layout="wide")
    
    st.title("🌿 Small Neural Network (NN) Iris Classifier Frontend")
    st.markdown("A demonstration of a trained $\\approx 100$ connection NN running in a web application.")
    
    if MODEL is None:
        st.stop() # Stop the app if resources couldn't be loaded

    # Sidebar for Model Info and instructions
    st.sidebar.header("Model Info")
    st.sidebar.markdown(f"**Model Path:** `{BEST_MODEL_PATH_BASE}`")
    st.sidebar.markdown(f"**Target Classes:** {', '.join(CLASS_NAMES)}")
    st.sidebar.markdown("**Architecture:** Dense Layers (Trained via Grid Search)")
    st.sidebar.divider()
    
    # --- Input Section ---
    st.header("1. Enter Flower Features")
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider('Sepal Length (cm)', 
                                 min_value=4.0, max_value=8.0, value=5.1, step=0.1)
        sepal_width = st.slider('Sepal Width (cm)', 
                                min_value=2.0, max_value=5.0, value=3.5, step=0.1)
    
    with col2:
        petal_length = st.slider('Petal Length (cm)', 
                                 min_value=1.0, max_value=7.0, value=1.4, step=0.1)
        petal_width = st.slider('Petal Width (cm)', 
                                min_value=0.1, max_value=3.0, value=0.2, step=0.1)

    input_data = [sepal_length, sepal_width, petal_length, petal_width]
    st.info(f"Current Input Data: {input_data}")

    st.divider()

    # --- Prediction Button and Output ---
    st.header("2. Run Prediction")

    if st.button('Classify Iris Species', type="primary"):
        with st.spinner('Running neural network prediction...'):
            predicted_class, probabilities = make_prediction(MODEL, SCALER, input_data)
            
            st.success(f"**Predicted Species:** {predicted_class.capitalize()}")
            st.balloons()
            
            st.subheader("Confidence Scores (Softmax Output)")
            
            # Display probabilities in a structured way
            prob_df = {
                "Species": [name.capitalize() for name in CLASS_NAMES],
                "Confidence": [f"{p*100:.2f}%" for p in probabilities]
            }
            
            st.table(prob_df)
            
            # Highlight the highest confidence
            highest_prob_index = np.argmax(probabilities)
            highest_prob_name = CLASS_NAMES[highest_prob_index].capitalize()
            highest_prob_value = probabilities[highest_prob_index] * 100
            
            st.metric(label="Highest Confidence", value=f"{highest_prob_value:.2f}%", help=f"Predicted as {highest_prob_name}")
            
if __name__ == '__main__':
    main()
