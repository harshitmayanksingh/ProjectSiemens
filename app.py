import numpy as np
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

model_path = "C:\SurfaceDefect\model_1.h5"
model = keras.models.load_model(model_path)


target_labels = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

def get_prediction_on_input_image(custom_image_path, model=model):
    # Open and preprocess the image
    custom_image = Image.open(custom_image_path).convert('L')
    custom_image = custom_image.convert('RGB')
    custom_image = custom_image.resize((200, 200))
    custom_image = np.array(custom_image)
    
    
    custom_image = custom_image.astype('float32') / 255.0
    
    custom_image = np.expand_dims(custom_image, axis=-1) #adding a dimension to image
    custom_image = np.expand_dims(custom_image, axis=0)
    
 
    predictions = model.predict(custom_image)
    
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = target_labels[predicted_class_index]

    plt.imshow(np.squeeze(custom_image), cmap='gray')
    plt.title(f"Predicted class: {predicted_class}")
    plt.axis('off')
    plt.show()


custom_image_path = r'C:\SurfaceDefect\Testing Images\testing\scratch1.png'

get_prediction_on_input_image(custom_image_path)
