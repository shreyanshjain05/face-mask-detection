import numpy as np
import streamlit as st
import os
from PIL import Image
from tensorflow.keras.models import load_model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f'{working_dir}/trained_model/model.h5'
model = load_model(model_path)
def image_preprocess(input_image_path):
    input_image = Image.open(input_image_path)
    image_resize = input_image.resize((128,128))
    image_scaled = np.array(image_resize)/255.0
    input_image_reshaped = np.reshape(image_scaled, [1, 128, 128, 3])
    return input_image_reshaped
def predict(image):
    prediction = model.predict(image)
    prediction = np.argmax(prediction)
    if prediction == 1:
        return 'Mask traced!'
    else:
        return 'No mask traced!'

st.title('Face Mask Detection system')
uploaded_image = st.file_uploader(label='Upload a JPG , PNG , BMP or JPEG image' , type = ['JPEG' , 'BMP' , 'JPG' , 'PNG'])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1 , col2 = st.columns(2)

    with col1:
        st.image(image , caption='Uploaded Image')
    with col2:
        if st.button('Classify'):
            preprocess_image = image_preprocess(uploaded_image)
            prediction = predict(preprocess_image)
            st.success(f'Prediction {prediction}' )