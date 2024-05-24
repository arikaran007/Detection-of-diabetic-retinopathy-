import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("CNN_model.h5")
    return model

model = load_model()

def predict_class(image):
    try:
       
        RGBImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        RGBImg = cv2.resize(RGBImg, (224, 224))
        image = np.array(RGBImg) / 255.0
        image = np.expand_dims(image, axis=0)
        predict = model.predict(image)
        per = np.argmax(predict, axis=1)
        return 'Diabetic Retinopathy Detected' if per == 0 else 'Diabetic Retinopathy Not Detected'
    except Exception as e:
        return f"Error processing image: {e}"


st.title("Diabetic Retinopathy Detection")
st.write("Upload an eye image to check for diabetic retinopathy.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
      
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")


        image = np.array(image)
       
        
        opencv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if opencv_image is not None:
           
            result = predict_class(opencv_image)
            st.write(result)
        else:
            st.write("Error: Could not load the image. Please try again.")
    except Exception as e:
        st.write(f"Error: {e}")
