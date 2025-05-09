Flask-Based Image Denoising App
Overview
This project is a Flask-based web application for image denoising using an autoencoder neural network. It includes a graphical user interface (GUI) to allow users to upload noisy images and receive denoised outputs. The system leverages a pre-trained autoencoder model to remove noise while preserving image details, and it is built with Flask for the backend and a custom GUI for user interaction.
Features

Upload noisy images via a web interface.
Denoise images using a pre-trained autoencoder model.
View and download the denoised results.
Lightweight and easy-to-use GUI for practical application.

Requirements
To run this project, ensure you have the following installed:

Python 3.8 or higher
Dependencies listed in requirements.txt

Setup Instructions

Clone the Repository  
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name


Create a Virtual Environment  
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies  
pip install -r requirements.txt


Run the Application  
python app.py

The app will start on http://localhost:5000. Open this URL in your browser to access the GUI.


Usage

Navigate to http://localhost:5000 in your browser.
Use the GUI to upload a noisy image.
The app will process the image using the autoencoder and display the denoised result.
Download the denoised image if needed.

Project Structure

app.py: Main Flask application file.
denoiser_gui.py: GUI implementation for the denoising functionality.
test_load.py: Script for testing or loading the model.
static/: Directory for static files (e.g., CSS, JavaScript).
templates/: Directory for HTML templates.
utils/: Utility scripts for the application.
requirements.txt: List of dependencies.

Dependencies
Key dependencies include:

Flask: Web framework for the backend.
TensorFlow/Keras: For the autoencoder model.
(Other dependencies as listed in requirements.txt)

Notes

Ensure the pre-trained autoencoder model is available and properly configured in denoiser_gui.py or related scripts.
The GUI is designed for ease of use but can be extended with additional features like batch processing.

Future Improvements

Add support for batch image processing.
Enhance the GUI with more interactive features.
Optimize the model for faster inference.

License
This project is licensed under the MIT License.
