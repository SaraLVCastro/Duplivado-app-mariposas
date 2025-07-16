import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

# Cargar el modelo YOLOv8 entrenado
model = YOLO('best.pt')

st.set_page_config(page_title="Detector de partes de mariposas", layout="centered")

st.title("🦋 Detector de partes de mariposas")
st.write("Sube una imagen de una mariposa y detectaremos sus partes.")

uploaded_file = st.file_uploader("Sube tu imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    with st.spinner("Detectando partes..."):
        results = model.predict(image)

    for r in results:
        annotated_img = r.plot()
        st.image(annotated_img, caption="Resultado", use_column_width=True)
