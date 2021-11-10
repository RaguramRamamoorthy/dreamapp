import streamlit as st
from PIL import Image, ImageOps
from img_classification import teachable_machine_classification


st.title("Brain Tumor MRI Classification")

st.text("Upload a brain MRI Image here to find weather it is healthy or not")

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

st.text("If you need MRI images for testing the App")
st.text("download a zip folder from here...")

with open("images.zip", "rb") as fp:
    btn = st.download_button(
        label="Download ZIP",
        data=fp,
        file_name="images.zip",
        mime="application/zip"
    )

st.text("If you need assistance in deploying a healthcare based machine learning or a deep")
st.text("learning model that might be useful for the general population or a specific")
st.text("hospital please feel free to reach out to me at 'rrrtechie@gmail.com'")\




