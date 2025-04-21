import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import torch
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.title("Alzheimer's Detection App")

# Upload image and show
test_image = st.file_uploader("Upload test image", type=["jpg", "jpeg"])

# Let the user select the model
model_names = {
    'Model 1: CNN ': 'modelsORweights/saved_model.keras',
    'Model 2: Vision Transformer': 'modelsORweights/VisTran with xTransformer.pth',
}

model_option = st.selectbox('Choose the model for prediction:', options=list(model_names.keys()))

# Define a function to load the model, this will use Streamlit's newer caching mechanism
@st.cache_resource
def load_model_wrapper(model_filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_filename.endswith('.pth'):
        model = torch.load(model_filename, map_location= device)
    else:
        model = load_model(model_filename)
    return model

if test_image is not None:
    # Display the uploaded image
    uploaded_image = Image.open(test_image).convert('RGB')
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for the model
    # img = uploaded_image.convert('RGB')
    # img = uploaded_image.resize((224, 224))

    img_array = image.img_to_array(uploaded_image)
    img_array = np.expand_dims(img_array, axis=0)
    
    #img_array = img_array / 255.0  # Normalize the image array

    # Load the selected model (only once per session)
    model_filename = model_names[model_option]
    model_selected = load_model_wrapper(model_filename)

    if model_filename.endswith('.keras'):
        # Make a prediction
        prediction = model_selected.predict(img_array)


    class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']

    # Convert the prediction to a percentage
    results = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}

    # Show as a table
    st.subheader("Prediction Results")
    st.table(results)
    

