
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

# Cargar el modelo YOLOv8 entrenado
model = YOLO('best.pt')

st.set_page_config(page_title="Detector de partes de mariposas", layout="centered")

st.title("🦋 Detector de partes de mariposas")
st.write("Sube una imagen de una mariposa y detectaremos sus partes.")

# Diccionario de relaciones anatómicas
relaciones = {
    "marginal band": "fore wing",
    "postapical spot band": "fore wing",
    "posterior discal spot": "fore wing",
    "apical band": "fore wing",
    "discal spot": "fore wing",
    "subapical spot": "fore wing",
    "anterior discal spot": "fore wing",
    "costal spot": "fore wing",
    "submarginal band": "fore wing",
    "subdiscal spot": "fore wing",
    "postdiscal spot band": "fore wing",
    "discal band": "fore wing",
    "postdiscal band": "fore wing",
    "subdiscal band": "fore wing",
    "subapical band": "fore wing",
    "marg": "fore wing",
    "anal spot": "fore wing",
    "tornal spot": "fore wing",
    "basal spot": "fore wing",
    "apical spot": "fore wing",
    "submarginal eye-spot": "fore wing",
    "postdiscal eye-spot": "fore wing",
    "postdiscal eye-spot band": "fore wing",
    "apical eye-spot": "fore wing",
    "marginal eye-spot": "fore wing",
    "posterior subdiscal spot": "fore wing",
    "anterior discal eye-spot": "fore wing",
    "posterior discal eye-spot": "fore wing",
    "dorsal spot": "fore wing",
    "subdiscal eye-spot": "fore wing",
    "eye-spot": "fore wing",
    "submarginal spot": "fore wing",
    "marginal spot": "fore wing"
}

uploaded_file = st.file_uploader("Sube tu imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    with st.spinner("Detectando partes..."):
        results = model.predict(image)

    for r in results:
        annotated_img = r.plot()
        st.image(annotated_img, caption="Resultado", use_column_width=True)

        etiquetas_detectadas = list(set([model.names[int(cls)] for cls in r.boxes.cls]))

        st.subheader("Descripción anatómica:")
        frases = []
        for etiqueta in etiquetas_detectadas:
            estructura = relaciones.get(etiqueta.lower())
            if estructura:
                frases.append(f"{etiqueta} is part of the {estructura}")
            else:
                frases.append(f"{etiqueta} is a main structure")
        descripcion = ". ".join(frases) + "."
        st.markdown(descripcion)
