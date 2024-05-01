import numpy as np
import time
from tensorflow.keras.preprocessing import image
import streamlit as st
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
# with tf.device('/cpu:0'):
# Load the saved model
model = tf.keras.models.load_model('best_resnet152_model.h5')

class_names = {0: '1099_Div', 1: '1099_Int', 2: 'Non_Form', 3: 'w_2', 4: 'w_3'}
# print(class_names)

# Load and preprocess the image
# img_path = '/app/filled_form_1.jpg'
@st.cache_resource
def predict(pil_img):
    # Convert the PIL image to a NumPy array
    img_array = image.img_to_array(pil_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale pixel values

    # Predict the class
    # with tf.device('/cpu:0'):
    start_time = time.time()
    predictions = model.predict(img_array)
    end_time = time.time()
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_index]
    print("Predicted class:", predicted_class_name)
    print("Execution time: ", end_time - start_time)
    return predicted_class_name