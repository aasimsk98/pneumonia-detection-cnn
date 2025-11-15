import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import pandas as pd 

st.set_page_config(
    page_title="Pneumonia Detection App",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_our_model():
    model_path = 'pneumonia_model.h5' 
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model

model = load_our_model()

def classify_image(image, model):
    image = ImageOps.fit(image, (150, 150), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = image_array.astype(np.float32) / 255.0
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction_raw = model.predict(data)
    score = prediction_raw[0][0]

    if score > 0.5:
        class_name = "PNEUMONIA"
        confidence = score
    else:
        class_name = "NORMAL"
        confidence = 1 - score
        
    return class_name, confidence

st.sidebar.title("Navigation")
page_selection = st.sidebar.selectbox("Go to:",["Home","Pneumonia Prediction", "Model Performance"])


if page_selection == "Home":
    st.title("Pneumonia Detection from Chest X-Rays")
    st.write("""
    Welcome to the Pneumonia Detection App using Chest X-Rays! 
    This app is built to classify chest X-ray images as either 'Normal' or 'Pneumonia' using a trained convolutional neural network.
    
    ### How it works:
    - **Upload**: Upload a chest X-ray image and the model will predict whether it is normal or has pneumonia.
    - **Accuracy**: The model provides a confidence score along with the classification.
    
    Please use the sidebar to navigate to different sections.
    """)

elif page_selection == "Pneumonia Prediction":
    
    st.title("Pneumonia Detection from Chest X-Rays")
    st.write("""
    This application uses a Convolutional Neural Network (CNN) built from scratch 
    to classify chest X-ray images as **Normal** or showing signs of **Pneumonia**. 
    
    Upload an image, and the model will analyze it.
    """)
    st.info("DISCLAIMER: This project is not a substitute for professional medical advice.")

    st.header("Upload a Chest X-ray Image")
    uploaded_file = st.file_uploader("", type=["jpeg", "jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded X-ray", use_container_width=True)
        
        with st.spinner('Analyzing the image...'):
            class_name, confidence = classify_image(image, model)
        
        with col2:
            st.subheader("Analysis Results:")
            if class_name == "PNEUMONIA":
                st.error(f"**Prediction:** {class_name}")
            else:
                st.success(f"**Prediction:** {class_name}")
                
            st.subheader("Confidence Score:")
            st.progress(float(confidence))
            st.metric(label="Confidence:", value=f"{confidence * 100:.2f}%")

elif page_selection == "Model Performance":
    
    st.title("Model Performance Evaluation")
    st.write("""
    This page shows the final report card for the model we are using. 
    This evaluation was performed on a test set of 624 images that the model had never seen before.
    """)

    st.header("Final Accuracy")
    st.metric(label="Accuracy on Test Images", value="90.22%")
    st.progress(0.9022)

    st.markdown("---")

    st.header("Classification Report")
    st.write("This report shows more about the model's performance.")
    
    report_data = {
        'Class': ['NORMAL', 'PNEUMONIA'],
        'Precision': [0.89, 0.91],
        'Recall': [0.85, 0.94],
        'F1-Score': [0.87, 0.92]
    }
    report_df = pd.DataFrame(report_data)
    st.table(report_df.set_index('Class'))
    
    with st.expander("What do these terms mean?"):
        st.write("""
        - **Precision:** When the model predicts a class, how often is it correct?" 
        
          (*When it predicts PNEUMONIA, it's right 91% of the time.*)
        
        - **Recall:** "Of all the actual cases of a class, how many did the model find?" 
        
          (*It successfully found 94% of all PNEUMONIA cases in the test set.*)
        """)

    st.markdown("---")

    st.header("Confusion Matrix")
    st.write("The confusion matrix shows the raw counts of correct and incorrect predictions.")

    matrix_data = {
        '': ['True NORMAL', 'True PNEUMONIA'],
        'Predicted NORMAL': [198, 25],
        'Predicted PNEUMONIA': [36, 365]
    }
    matrix_df = pd.DataFrame(matrix_data)
    st.table(matrix_df.set_index(''))
    
    with st.expander("How to read this matrix"):
        st.write("""
        - **Top-Left (198):** Correctly identified 198 NORMAL patients (True Negatives).
        - **Bottom-Right (365):** Correctly identified 365 PNEUMONIA patients (True Positives).
        - **Top-Right (36):** Incorrectly flagged 36 NORMAL patients as having pneumonia (False Positives).
        - **Bottom-Left (25):** Dangerously missed 25 PNEUMONIA patients, labeling them as NORMAL (False Negatives).
        """)

st.markdown("---")
st.markdown("**Github Link**: ")