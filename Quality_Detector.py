from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Function to preprocess image data
def preprocess_image(img_path, img_height, img_width):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# Load the trained model
model_path = '/content/drive/MyDrive/TRAINED MODEL 50/'
model = tf.keras.models.load_model(model_path)

# Image dimensions expected by the model
img_height = 256
img_width = 256

# Preprocess the given data
given_data_path = '/content/drive/MyDrive/Dataset/validation/defected/leg_broken_158.jpg'
preprocessed_data = preprocess_image(given_data_path, img_height, img_width)

# Make predictions
prediction = model.predict(preprocessed_data)

# Interpret predictions
if prediction < 0.5:
    print("The given data is defected.")
else:
    print("The given data is undefected.")