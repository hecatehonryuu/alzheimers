import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import torch
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from vit_pytorch.efficient import ViT
from x_transformers import Encoder
from torchvision import transforms

st.title("Alzheimer's Detection App")

# Upload image and show
test_image = st.file_uploader("Upload test image", type=["jpg", "jpeg"])

# Let the user select the model
model_names = {
    'Model 1: Keras CNN ': 'modelsORweights/saved_model.keras',
    'Model 2: Vision Transformer Pytorch': 'modelsORweights/VisTran with xTransformer.pth',
}

model_option = st.selectbox('Choose the model for prediction:', options=list(model_names.keys()))

# Define a function to load the model, this will use Streamlit's caching mechanism
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

    # Load the selected model (only once per session)
    model_filename = model_names[model_option]
    model_selected = load_model_wrapper(model_filename)

    if model_filename.endswith('.keras'):
        # Make a prediction
        img_array = uploaded_image.resize((208, 176))
        img_array = image.img_to_array(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model_selected.predict(img_array)

        class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']

        # Convert the prediction to a percentage
        results = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}

        # Show as a table
        st.subheader("Prediction Results")
        st.table(results)

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #Importing base model
        efficient_transformer = Encoder(
            dim = 128,                  
            depth = 12,
            heads = 8,
            ff_glu = True,              
            residual_attn = True   
        )
        model1 = ViT(
            dim=128,
            image_size=224,
            patch_size=32,
            num_classes=2,
            transformer=efficient_transformer,
            channels=1,
        )

        model1.load_state_dict(model_selected)
        model1.to(device)
        model1.eval()

        uploaded_image = uploaded_image.convert("L")

        transform1 = transforms.Compose([
            transforms.Resize((256, 256)),  # Change size if needed
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # Convert to tensor
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Use dataset-specific values
            #                std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform1(uploaded_image).unsqueeze(0).to(device)  # Add batch dimension
        
        class_names = ["Normal - No Alzheimer's Detected", "Alzheimer Detected"]

        # Predict for pytorch model
        with torch.no_grad():
            output = model1(input_tensor)
            prediction = torch.argmax(output, dim=1)
            predicted_index = prediction.item()
            predicted_class = class_names[predicted_index]
            st.success(f"Prediction: {predicted_class}")

    
    

