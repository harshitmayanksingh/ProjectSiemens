import numpy as np
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained model
model_path = 'C:\SurfaceDefect\model_1.h5'
model = keras.models.load_model(model_path)

# List of class labels. Replace with your own class labels in the correct order.
target_labels = ["Inclusion", "Patches", "Rolled", "Pitted", "Scratches", "Crazing"]

def get_prediction_on_input_image(custom_image_path, model=model):
    # Open and preprocess the image
    custom_image = Image.open(custom_image_path).convert('RGB')
    custom_image = custom_image.resize((200, 200))
    custom_image = np.array(custom_image)
    
    custom_image = custom_image.astype('float32') / 255.0
    custom_image = np.expand_dims(custom_image, axis=0)
    
    predictions = model.predict(custom_image)
    
    # Decode prediction to class label and display it
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = target_labels[predicted_class_index]

    plt.imshow(np.squeeze(custom_image))
    plt.title(f"Predicted class: {predicted_class}")  # Fixed this line
    plt.axis('off')
    plt.show()


custom_image_path = 'C:\\SurfaceDefect\\testing1.png'
  # Replace with your test image's path
get_prediction_on_input_image(custom_image_path)
