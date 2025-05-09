import tensorflow as tf
from tensorflow import keras
import os
import numpy as np # Often needed implicitly by TF/Keras

# --- Configuration ---
# !!! IMPORTANT: Use the SAME LOCAL PATH as in your GUI script !!!
MODEL_PATH = r'model\best_denoising_autoencoder.keras'
# Or use the full path if preferred:
# MODEL_PATH = r'F:\KDU\port\py\model\denoiser_unet_v3_best.keras'

# --- Custom Loss Function Definition (Must match definition used for saving/in GUI) ---
# !! Make sure this definition is correct !!
PERCEPTUAL_LOSS_ALPHA = 0.8 # Value used during training
PERCEPTUAL_LOSS_BETA = 0.2  # Value used during training
def perceptual_loss(y_true, y_pred):
    pixel_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    # Dummy feature loss if VGG isn't needed/available for loading structure
    feature_loss = tf.constant(0.0, dtype=tf.float32)
    total_loss = PERCEPTUAL_LOSS_ALPHA * pixel_loss + PERCEPTUAL_LOSS_BETA * feature_loss
    return total_loss

custom_objects = {'perceptual_loss': perceptual_loss}

print(f"--- Attempting to load model: {MODEL_PATH} ---")

if not os.path.exists(MODEL_PATH):
    print(f"!!! ERROR: Model file not found at the specified path !!!")
else:
    try:
        # Attempt to load the model
        loaded_model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        print("\n--- SUCCESS: Model loaded successfully! ---")
        print("Model Summary:")
        loaded_model.summary() # Print summary if loaded

    except Exception as e:
        print(f"\n!!! ERROR: Failed to load model !!!")
        print(f"Specific Error: {e}")
        import traceback
        print("\n--- Full Traceback ---")
        traceback.print_exc() # Print the full error details
        print("--- End Traceback ---")

print("\n--- Test script finished ---")