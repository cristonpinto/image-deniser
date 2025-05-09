# denoiser_gui.py
# A GUI script to test the denoising model with a front-end interface
# Modified to support patch-based denoising

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import time

# Define paths
MODEL_PATH = "best_finetuned_model.keras"  # Update to "best_finetuned_model.h5" if you fine-tuned

# Load the pre-trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Ensure the model file is in the root directory.")

print(f"Loading pre-trained model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Noise addition function
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

# Original denoising function (processes whole image at once)
def denoise_image(model, noisy_image):
    if noisy_image.ndim == 3:
        noisy_image = np.expand_dims(noisy_image, axis=0)
    denoised_image = model.predict(noisy_image, batch_size=1)
    return denoised_image[0]

# New patch-based denoising function
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

# Metrics calculation
def calculate_metrics(original, denoised):
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

# GUI Class
class DenoiserGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Patch-Based Image Denoiser GUI")
        self.root.geometry("1200x650")

        # Variables
        self.image_path = None
        self.original_img = None
        self.noisy_img = None
        self.denoised_img = None
        self.noise_type = tk.StringVar(value="gaussian")
        self.noise_level = tk.DoubleVar(value=0.1)
        self.use_patches = tk.BooleanVar(value=True)
        self.patch_size = tk.IntVar(value=128)
        self.patch_overlap = tk.IntVar(value=16)
        self.processing = False

        # GUI Elements
        # Image Panels
        self.panel_original = tk.Label(self.root, text="Original Image")
        self.panel_original.grid(row=0, column=0, padx=10, pady=5)
        self.panel_noisy = tk.Label(self.root, text="Noisy Image")
        self.panel_noisy.grid(row=0, column=1, padx=10, pady=5)
        self.panel_denoised = tk.Label(self.root, text="Denoised Image")
        self.panel_denoised.grid(row=0, column=2, padx=10, pady=5)

        # Noise Options
        self.noise_frame = tk.Frame(self.root)
        self.noise_frame.grid(row=1, column=0, columnspan=3, pady=10)

        tk.Label(self.noise_frame, text="Noise Type:").pack(side=tk.LEFT, padx=5)
        noise_types = ['gaussian', 'salt_pepper', 'speckle']
        self.noise_menu = ttk.Combobox(self.noise_frame, textvariable=self.noise_type, values=noise_types, state="readonly")
        self.noise_menu.pack(side=tk.LEFT, padx=5)

        tk.Label(self.noise_frame, text="Noise Level (0.05-0.5):").pack(side=tk.LEFT, padx=5)
        self.noise_slider = tk.Scale(self.noise_frame, from_=0.05, to=0.5, resolution=0.05, orient=tk.HORIZONTAL, variable=self.noise_level)
        self.noise_slider.pack(side=tk.LEFT, padx=5)

        # Patch Options
        self.patch_frame = tk.Frame(self.root)
        self.patch_frame.grid(row=2, column=0, columnspan=3, pady=10)

        self.patch_check = tk.Checkbutton(self.patch_frame, text="Use Patch-based Processing", variable=self.use_patches)
        self.patch_check.pack(side=tk.LEFT, padx=5)

        tk.Label(self.patch_frame, text="Patch Size:").pack(side=tk.LEFT, padx=5)
        patch_sizes = [64, 96, 128, 192, 256]
        self.patch_size_menu = ttk.Combobox(self.patch_frame, textvariable=self.patch_size, values=patch_sizes, state="readonly", width=5)
        self.patch_size_menu.pack(side=tk.LEFT, padx=5)

        tk.Label(self.patch_frame, text="Overlap:").pack(side=tk.LEFT, padx=5)
        overlap_sizes = [0, 8, 16, 24, 32, 48]
        self.overlap_menu = ttk.Combobox(self.patch_frame, textvariable=self.patch_overlap, values=overlap_sizes, state="readonly", width=5)
        self.overlap_menu.pack(side=tk.LEFT, padx=5)

        # Buttons
        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.grid(row=3, column=0, columnspan=3, pady=10)

        self.btn_load = tk.Button(self.btn_frame, text="Load Image", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=20)

        self.btn_denoise = tk.Button(self.btn_frame, text="Add Noise & Denoise", command=self.process_image)
        self.btn_denoise.pack(side=tk.LEFT, padx=20)

        self.btn_clear = tk.Button(self.btn_frame, text="Clear", command=self.clear_images)
        self.btn_clear.pack(side=tk.LEFT, padx=20)

        # Status and Metrics Label
        self.status_label = tk.Label(self.root, text="Status: Ready")
        self.status_label.grid(row=4, column=0, columnspan=3, pady=5)

        self.metrics_label = tk.Label(self.root, text="PSNR: N/A, SSIM: N/A")
        self.metrics_label.grid(row=5, column=0, columnspan=3, pady=5)

        # Image Info Label
        self.image_info_label = tk.Label(self.root, text="No image loaded")
        self.image_info_label.grid(row=6, column=0, columnspan=3, pady=5)

    def load_image(self):
        """Load an image from the local machine"""
        if self.processing:
            return
            
        self.image_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if self.image_path:
            # Load the image
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # If using patch-based processing, keep original size (with limits)
            # Otherwise, resize to model input size
            if self.use_patches.get():
                max_dim = 1024  # Set a reasonable maximum dimension
                h, w = img.shape[:2]
                if h > max_dim or w > max_dim:
                    scale = max_dim / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                self.original_img = img.astype('float32') / 255.0
                self.image_info_label.config(text=f"Image loaded: {os.path.basename(self.image_path)} - Size: {self.original_img.shape[1]}x{self.original_img.shape[0]}")
            else:
                img_resized = cv2.resize(img, (128, 128))  # Resize to model input size
                self.original_img = img_resized.astype('float32') / 255.0
                self.image_info_label.config(text=f"Image loaded: {os.path.basename(self.image_path)} - Resized to: 128x128")

            # Display the original image
            self.display_image(self.original_img, self.panel_original)
            self.metrics_label.config(text="PSNR: N/A, SSIM: N/A")
            self.panel_noisy.config(image='')
            self.panel_denoised.config(image='')
            self.status_label.config(text="Status: Image loaded successfully")

    def process_image(self):
        """Add noise and denoise the image"""
        if self.original_img is None:
            messagebox.showerror("Error", "Please load an image first!")
            return
            
        if self.processing:
            messagebox.showinfo("Processing", "Already processing an image. Please wait.")
            return
            
        self.processing = True
        self.status_label.config(text="Status: Processing image...")
        self.root.update()
        
        try:
            # Add noise
            noise_type = self.noise_type.get()
            noise_level = self.noise_level.get()
            self.noisy_img = add_noise_tf(self.original_img, noise_type=noise_type, noise_level=noise_level).numpy()

            # Denoise the image
            start_time = time.time()
            
            if self.use_patches.get():
                patch_size = self.patch_size.get()
                overlap = self.patch_overlap.get()
                self.status_label.config(text=f"Status: Denoising with patch size {patch_size}x{patch_size}, overlap {overlap}px...")
                self.root.update()
                self.denoised_img = denoise_image_patches(model, self.noisy_img, patch_size=patch_size, overlap=overlap)
            else:
                self.status_label.config(text="Status: Denoising whole image...")
                self.root.update()
                self.denoised_img = denoise_image(model, self.noisy_img)
                
            elapsed_time = time.time() - start_time

            # Calculate metrics
            psnr_val, ssim_val = calculate_metrics(self.original_img, self.denoised_img)
            self.metrics_label.config(text=f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f} - Processing time: {elapsed_time:.2f}s")

            # Display noisy and denoised images
            self.display_image(self.noisy_img, self.panel_noisy)
            self.display_image(self.denoised_img, self.panel_denoised)
            
            self.status_label.config(text="Status: Processing completed")
        except Exception as e:
            self.status_label.config(text=f"Status: Error - {str(e)}")
            messagebox.showerror("Error", f"An error occurred during processing: {str(e)}")
        finally:
            self.processing = False

    def display_image(self, img_array, panel):
        """Display an image in the specified panel"""
        # Convert to float32 for compatibility with PIL
        img_array = img_array.astype(np.float32)
        # Convert to uint8 for display (scale from [0,1] to [0,255])
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        # Resize for display (optional, adjust size as needed)
        img = img.resize((300, 300), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        panel.config(image=photo)
        panel.image = photo  # Keep a reference to avoid garbage collection

    def clear_images(self):
        """Clear all images and reset the GUI"""
        if self.processing:
            return
            
        self.original_img = None
        self.noisy_img = None
        self.denoised_img = None
        self.panel_original.config(image='')
        self.panel_noisy.config(image='')
        self.panel_denoised.config(image='')
        self.metrics_label.config(text="PSNR: N/A, SSIM: N/A")
        self.image_info_label.config(text="No image loaded")
        self.status_label.config(text="Status: Ready")

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = DenoiserGUI(root)
    root.mainloop()