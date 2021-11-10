import streamlit as st
from PIL import Image, ImageOps
from img_classification import teachable_machine_classification
import os



st.title("Brain Tumor MRI Classification")


st.text("If you need MRI images for testing the App")
st.text("Copy paste this link in a browser ")
st.text("https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection")


# folder = os.getcwd() + '/images'
# st.download_button('Download images', folder)

with open("images.zip", "rb") as fp:
    btn = st.download_button(
        label="Download ZIP",
        data=fp,
        file_name="images.zip",
        mime="application/zip"
    )

st.text("Upload a brain MRI Image to find weather it is healthy or not")

uploaded_file = st.file_uploader("Choose a brain MRI ...",type=["png","jpg","jpeg"])
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


st.text("If you need assistance in deploying a medicine based machine learning or a deep")
st.text("learning model that might be useful for the general popualtion or a specific")
st.text("hospital please reach out to us")
st.text("Contact me at 'rrrtechie@gmail.com'")

