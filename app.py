import streamlit as st
from PIL import Image, ImageOps
from img_classification import teachable_machine_classification
import webbrowser


st.title("Med App")
st.header("Brain Tumor MRI Classification Example")

url = 'https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection'

st.text("If you need MRI images for testing the app ,please find it in the kaggle link below")

if st.button('Images-link'):
    webbrowser.open_new_tab(url)

st.text("Upload a brain MRI Image for image classification as tumor or no-tumor")

uploaded_file = st.file_uploader("Choose a brain MRI ...")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'model.h5')
    if label == 0:
        st.write("The MRI scan is healthy")
    else:
        st.write("The MRI scan has a brain tumor")
