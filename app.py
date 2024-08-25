# ---------------------------

import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO

# Load your YOLO model (assuming it's a PyTorch model)
try:
    model = YOLO(r'C:\Users\ASUS\PycharmProjects\isro_fronte\isro.v3-001\isro.v3\runs\detect\train2\weights\final_verion6_best.pt')
    st.sidebar.success('Model loaded successfully!')
except Exception as e:
    st.sidebar.error('Error loading model')
    st.sidebar.error(e)

# Streamlit sidebar for user input
st.sidebar.title('Feature extraction and Object Detection')
st.sidebar.write('Upload an image to perform object detection')

# Main screen for image upload and prediction
st.title('Feature Extraction and Detection using high resolution aerial images')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Save the uploaded image to a temporary file
        temp_image_path = 'temp_image.png'
        image = Image.open(uploaded_file)
        image.save(temp_image_path)

        # Display the uploaded image and prediction result side by side
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Predict'):
            with st.spinner('Predicting...'):
                # Perform the prediction
                results = model.predict(source=temp_image_path, conf=0.25, save=True)

                # Find the saved result image
                saved_dir = results[0].save_dir  # Directory where results are saved
                saved_image_path = os.path.join(saved_dir, os.path.basename(temp_image_path))

                # Display the predicted image side by side
                with col2:
                    result_image = Image.open(saved_image_path)
                    st.image(result_image, caption='Predicted Image', use_column_width=True)

                # st.write('Prediction Results:')
                # st.write(results[0].boxes)  # Display the prediction results as a table

                # Clean up temporary file
                os.remove(temp_image_path)

            # User feedback section
            st.title('User Feedback')
            st.write('Please provide your feedback on the prediction:')
            feedback = st.text_area('Feedback')
            if st.button('Submit Feedback'):
                st.write('Thank you for your feedback!')

    except Exception as e:
        st.error('Error during prediction')
        st.error(e)