import os
import time
import numpy as np
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def load_model(model_path):
    """Load the denoising model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Loading pre-trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def add_noise_tf(image, noise_type='gaussian', noise_level=0.1):
    """Add noise to the image using TensorFlow operations"""
    if noise_type == 'gaussian':
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=noise_level)
        noisy_image = image + noise
    elif noise_type == 'salt_pepper':
        s_vs_p = 0.5
        amount = noise_level
        salt_mask = tf.random.uniform(tf.shape(image)) < amount * s_vs_p
        pepper_mask = tf.random.uniform(tf.shape(image)) < amount * (1 - s_vs_p)
        noisy_image = tf.where(salt_mask, 1.0, image)
        noisy_image = tf.where(pepper_mask, 0.0, noisy_image)
    elif noise_type == 'speckle':
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=noise_level)
        noisy_image = image + image * noise
    else:
        noisy_image = image
    
    noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)
    return noisy_image

def denoise_image(model, noisy_image):
    """Process whole image at once"""
    if noisy_image.ndim == 3:
        noisy_image = np.expand_dims(noisy_image, axis=0)
    denoised_image = model.predict(noisy_image, batch_size=1)
    return denoised_image[0]

def denoise_image_patches(model, noisy_image, patch_size=128, overlap=16):
    """Denoise an image by processing it in overlapping patches"""
    h, w, c = noisy_image.shape
    
    # Calculate padding if needed to make dimensions divisible by (patch_size - overlap)
    step_size = patch_size - overlap
    pad_h = (step_size - h % step_size) % step_size if h % step_size != 0 else 0
    pad_w = (step_size - w % step_size) % step_size if w % step_size != 0 else 0
    
    if pad_h > 0 or pad_w > 0:
        # Pad the image to make it divisible by step_size
        padded_img = np.pad(noisy_image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        padded_img = noisy_image
    
    # Get new dimensions after padding
    h_padded, w_padded, _ = padded_img.shape
    
    # Initialize the output image
    denoised_padded = np.zeros_like(padded_img)
    # Initialize a weight map for patch blending
    weight_map = np.zeros_like(padded_img)
    
    # Create a simple weight map for blending (higher weights in the center of patches)
    patch_weight = np.ones((patch_size, patch_size, c))
    if overlap > 0:
        # Create a tapering effect at the edges for better blending
        for i in range(overlap):
            # Decrease weight at the edges
            factor = (i + 1) / (overlap + 1)
            patch_weight[i, :, :] *= factor
            patch_weight[patch_size-i-1, :, :] *= factor
            patch_weight[:, i, :] *= factor
            patch_weight[:, patch_size-i-1, :] *= factor
    
    # Process overlapping patches
    total_patches = ((h_padded - patch_size) // step_size + 1) * ((w_padded - patch_size) // step_size + 1)
    processed_patches = 0
    
    print(f"Processing image of size {h}x{w} in {total_patches} patches of size {patch_size}x{patch_size} with {overlap} pixel overlap")
    start_time = time.time()
    
    for y in range(0, h_padded - patch_size + 1, step_size):
        for x in range(0, w_padded - patch_size + 1, step_size):
            # Extract the patch
            patch = padded_img[y:y+patch_size, x:x+patch_size, :]
            
            # Denoise the patch
            patch = np.expand_dims(patch, axis=0)  # Add batch dimension
            denoised_patch = model.predict(patch, batch_size=1, verbose=0)[0]
            
            # Add the denoised patch to the output image with weighting
            denoised_padded[y:y+patch_size, x:x+patch_size, :] += denoised_patch * patch_weight
            weight_map[y:y+patch_size, x:x+patch_size, :] += patch_weight
            
            processed_patches += 1
            if processed_patches % 10 == 0 or processed_patches == total_patches:
                print(f"Processed {processed_patches}/{total_patches} patches")
    
    # Normalize by the weight map to blend patches smoothly
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    denoised_padded = denoised_padded / (weight_map + epsilon)
    
    # Crop back to original dimensions
    denoised_image = denoised_padded[:h, :w, :]
    
    elapsed_time = time.time() - start_time
    print(f"Denoising completed in {elapsed_time:.2f} seconds")
    
    return denoised_image

def calculate_metrics(original, denoised):
    """Calculate PSNR and SSIM metrics"""
    original = original.astype(np.float64)
    denoised = denoised.astype(np.float64)
    psnr_value = psnr(original, denoised, data_range=1.0)
    
    min_dim = min(original.shape[0], original.shape[1])
    if min_dim < 3:
        print(f"Warning: Image dimension ({min_dim}) too small for SSIM calculation.")
        win_size = min_dim if min_dim % 2 != 0 else max(1, min_dim - 1)
        if win_size == 0:
            win_size = 1
    elif min_dim < 7:
        win_size = min_dim if min_dim % 2 != 0 else min_dim - 1
    else:
        win_size = 7
    
    if original.ndim == 3:
        ssim_value = ssim(original, denoised, data_range=1.0, channel_axis=-1, win_size=win_size)
    else:
        raise ValueError(f"Unexpected image dimension: {original.ndim}")
    
    return psnr_value, ssim_value