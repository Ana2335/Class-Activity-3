import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image

# Subir modelos
def load_models():
    with open("cnn2.pkl", "rb") as f1, \
         open("vgg162.pkl", "rb") as f2, \
         open("mobilnet2.pkl", "rb") as f3:
        model_baseline = pickle.load(f1)
        model_vgg16 = pickle.load(f2)
        model_mobilenet = pickle.load(f3)
    return model_baseline, model_vgg16, model_mobilenet

model_baseline, model_vgg16, model_mobilenet = load_models()

etiquetas = ["Angular Leaf Spot", "Bean Rust", "Healthy"]

st.title("Bean leaf clasification 🌱🫘")
st.write("Upload an image to predict its category")
img = st.file_uploader("Enter an image", type=["jpg", "png", "jpeg"])

if img:
    # Cargar la imagen
    img = Image.open(img).convert("RGB")
    st.image(img, caption="Image uploaded", use_column_width=True)

    # Preprocesamiento 
    def preprocess(img, size):
        resized = img.resize(size)
        arr = np.array(resized) / 255.0
        return np.expand_dims(arr, axis=0)

    input_baseline = preprocess(img, (180, 180))
    input_vgg16 = preprocess(img, (180, 180))
    input_mobilenet = preprocess(img, (128, 128))

    # Predicciones
    if st.button("Predict"):
        pred_base = model_baseline.predict(input_baseline)[0]
        pred_vgg = model_vgg16.predict(input_vgg16)[0]
        pred_mobile = model_mobilenet.predict(input_mobilenet)[0]

        st.subheader("Results of each model")
        cols = st.columns(3)

        for i, (model_name, pred) in enumerate([
            ("Baseline CNN", pred_base),
            ("VGG16", pred_vgg),
            ("MobileNetV2", pred_mobile)
        ]):
            pred_class = etiquetas[np.argmax(pred)]
            with cols[i]:
                st.markdown(f"**{model_name}**")
                st.markdown(f"Prediction 🔍: **{pred_class}**")
