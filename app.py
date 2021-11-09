import streamlit as st
from PIL import Image, ImageOps
from img_classification import teachable_machine_classification
import webbrowser


st.title("Brain Tumor MRI Classification")

link = '[Kaggle-link for MRI images](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)'


st.text("If you need MRI images for testing the App,please find it in the kaggle link below")
st.markdown(link, unsafe_allow_html=True)



st.text("Upload a brain MRI Image for image classification to find weather the scan image")
st.text("is healthy or not")

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
