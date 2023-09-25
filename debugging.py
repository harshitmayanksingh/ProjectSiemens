import numpy as np
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt


model_path = 'C:\\SurfaceDefect\\model_1.h5'
model = keras.models.load_model(model_path)

# List of class labels. Replace with your own class labels in the correct order.
target_labels = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]

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
    
    print(f'Predicted class index: {predicted_class_index}')  # Debugging line
    print(f'Predicted class: {predicted_class}')  # Debugging line
    
    plt.imshow(np.squeeze(custom_image))
    plt.title(f'Predicted class: {predicted_class}')  # Fixed this line
    plt.axis('off')
    plt.show()

custom_image_path = 'Testing Images\custom_img_4_inclusion.jpeg'  # Using double backs
get_prediction_on_input_image(custom_image_path)