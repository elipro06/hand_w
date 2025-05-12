import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# --- Configuración de la página ---
st.set_page_config(page_title='✍️ Reconocimiento de Dígitos Escritos a Mano', layout='wide')

# --- Estilos personalizados ---
st.markdown("""
    <style>
    html, body, [data-testid="stApp"] {
        background-color: #dbeeff !important;
        color: #000000 !important;
    }
    .block-container {
        background-color: #dbeeff !important;
        color: #000000 !important;
    }
    h1, h2, h3, p, label, span, div {
        color: #000000 !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #4a90e2, #50a7f6);
        color: white !important;
        font-weight: bold;
        border-radius: 12px;
        height: 48px;
        font-size: 16px;
        margin-top: 10px;
    }
    .stSlider {
        color: #000000 !important;
    }
    .stSidebar {
        background-color: #d0e7ff !important;
        color: #000000 !important;
    }
    .css-1vq4p4l {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Título principal ---
st.title('🧠 Reconocimiento de Dígitos Escritos a Mano')
st.subheader("🖌️ Dibuja el dígito en el panel y presiona el botón 'Predecir' para ver el resultado")

# --- Parámetros del canvas ---
drawing_mode = "freedraw"
stroke_width = st.slider('✏️ Selecciona el grosor del trazo', 1, 30, 15)
stroke_color = '#FFFFFF'  # blanco
bg_color = '#000000'      # negro

# --- Área de dibujo (canvas) ---
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    drawing_mode=drawing_mode,
    key="canvas",
)

# --- Modelo de predicción ---
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32') / 255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

# --- Botón de predicción ---
if st.button('🔍 Predecir'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.header(f'✅ El dígito predicho es: **{res}**')
    else:
        st.warning('⚠️ Por favor dibuja un dígito antes de predecir.')

# --- Barra lateral ---
st.sidebar.title("ℹ️ Acerca de la App")
st.sidebar.markdown("""
Esta aplicación demuestra cómo una red neuronal 
puede reconocer dígitos escritos a mano ✍️🧠

📚 Basado en el trabajo de Vinay Uniyal  
📦 Modelo entrenado con MNIST
""")

