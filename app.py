# import streamlit as st
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import joblib

# # ================= PAGE CONFIG (MUST BE FIRST) =================
# st.set_page_config(
#     page_title="Food Image Classifier",
#     page_icon="🍔",
#     layout="centered"
# )

# # ================= LOAD MODEL & CLASSES =================
# @st.cache_resource
# def load_model_and_classes():
#     model = tf.keras.models.load_model("food_model.h5")
#     class_names = joblib.load("class_names.pkl")
#     return model, class_names

# model, class_names = load_model_and_classes()

# img_height = 128
# img_width = 128

# # ================= SIDEBAR =================
# st.sidebar.title("About App")
# st.sidebar.info(
#     "**Food Image Classifier**\n\n"
#     "Upload a food image\n\n"
#     "CNN-based AI model\n\n"
#     "Streamlit Web App"
# )

# # ================= TITLE =================
# st.markdown(
#     """
#     <h1 style='text-align: center; color: #FF5733;'>Food Image Classifier</h1>
#     <p style='text-align: center; font-size: 18px;'>
#     Upload a food image and let AI predict its category!
#     </p><hr>
#     """,
#     unsafe_allow_html=True
# )

# # ================= FILE UPLOAD =================
# st.subheader("Upload a Food Image")
# uploaded_file = st.file_uploader(
#     "Supported formats: JPG, JPEG, PNG",
#     type=["jpg", "jpeg", "png"]
# )

# # ================= MAIN LOGIC =================
# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")

#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # -------- PREPROCESS (MATCHES TRAINING) --------
#     image = image.resize((img_width, img_height))
#     image_array = tf.keras.preprocessing.image.img_to_array(image)
#     image_array = image_array / 255.0
#     image_array = np.expand_dims(image_array, axis=0)

#     # -------- PREDICT --------
#     with st.spinner("Analyzing image..."):
#         predictions = model.predict(image_array)

#     predicted_index = np.argmax(predictions[0])
#     predicted_label = class_names[predicted_index]
#     confidence = float(np.max(predictions[0]) * 100)

#     # -------- OUTPUT --------
#     st.success(
#         f"Prediction: {predicted_label}\n\n"
#         f"Confidence: {confidence:.2f}%"
#     )

#     # -------- OPTIONAL: TOP 3 PREDICTIONS --------
#     st.subheader("Top Predictions")
#     top3 = np.argsort(predictions[0])[-3:][::-1]
#     for i in top3:
#         st.write(f"{class_names[i]}: {predictions[0][i] * 100:.2f}%")


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Food Classification App",
    layout="centered"
)

# -------------Setting the Image here---------------
img_path = "Food.jpg"
if os.path.exists(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    st.markdown(f"""
    <div style="display:flex; justify-content:center; margin-bottom:20px;">
        <img src="data:image/png;base64,{b64}"
             style="
                width:1200px;
                height:200px;
                border:4px solid #5E8A8E;
                border-radius:20px;
                box-shadow:0 0 12px rgba(94,138,142,0.5);
                object-fit:fill;
             ">
    </div>
    """, unsafe_allow_html=True)

# ================= TITLE =================
st.markdown(
    """
    <h1 style='text-align: center; color: #FF5733;'>Food Image Classifier</h1>
    <p style='text-align: center; font-size: 18px;'>
    Upload a food image and let AI predict its category!
    </p><hr>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("food_model.h5")

model = load_model()

# ---------------- CLASS NAMES ----------------
# IMPORTANT: order must match training order
class_names = [
    "donuts",
    "french_fries",
    "hamburger",
    "hot_dog",
    "pizza",
    "samosa",
    "sushi",
    "waffles"
]

IMG_SIZE = (160, 160)

# ================= SIDEBAR =================
st.sidebar.title("About App")
st.sidebar.info(
    "**Food Image Classifier**\n\n"
    "Upload a food image\n\n"
    "CNN-based AI model\n\n"
    "Streamlit Web App"
)

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array  # NO /255 (already in model)

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=600)

    if st.button("Predict"):
        with st.spinner("Classifying..."):
            img_array = preprocess_image(image)
            predictions = model.predict(img_array)
            confidence = np.max(predictions)
            predicted_class = class_names[np.argmax(predictions)]

        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence*100:.2f}%")

        # Optional confidence warning
        if confidence < 0.5:
            st.warning("Low confidence prediction. Try a clearer image.")

# ================= FOOTER =================
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: gray;'>
    Developed with love using Streamlit 💖
    </p>
    """,
    unsafe_allow_html=True
)

