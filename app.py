import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Product Image Classifier",
    page_icon="",
    layout="centered"
)


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

MODEL_PATH = Path("models/product_classifier_v2_best.keras")

IMG_SIZE = (160, 160)

CLASS_NAMES = [
    "Tshirts",
    "Shirts",
    "Tops",
    "Dresses",
    "Casual Shoes",
    "Sports Shoes",
    "Heels",
    "Watches",
    "Handbags",
    "Sunglasses"
]


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_image(image, model, top_k=3):

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)

    # IMPORTANT:
    # MobileNetV2 preprocessing is already inside the model.
    predictions = model.predict(image_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "class": CLASS_NAMES[index],
            "confidence": float(predictions[index])
        })

    return results


# --------------------------------------------------
# MAIN INTERFACE
# --------------------------------------------------

st.title(" AI Product Image Classifier")

st.write(
    "Upload an image and let the AI model identify "
    "the most likely product category."
)

st.divider()


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a clothing or product item."
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Your uploaded image",
        width="stretch"
    )

    st.divider()

    with st.spinner("Analyzing image..."):

        results = predict_image(
            image,
            model,
            top_k=3
        )

    # --------------------------------------------------
    # TOP PREDICTION
    # --------------------------------------------------

    top_result = results[0]

    st.subheader("Prediction")

    st.success(
        f"**{top_result['class']}** "
        f"({top_result['confidence'] * 100:.2f}% confidence)"
    )

    st.divider()

    # --------------------------------------------------
    # TOP 3
    # --------------------------------------------------

    st.subheader("Top 3 Predictions")

    for i, result in enumerate(results):

        confidence = result["confidence"]

        st.write(
            f"**{i + 1}. {result['class']}** "
            f"— {confidence * 100:.2f}%"
        )

        st.progress(confidence)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("About the Model")

    st.write(
        "This application uses a MobileNetV2-based "
        "image classification model trained using "
        "transfer learning."
    )

    st.write("### Model Details")

    st.write("• Architecture: MobileNetV2")
    st.write("• Image Size: 160 × 160")
    st.write("• Number of Classes: 10")
    st.write("• Test Accuracy: 91.17%")
    st.write("• Macro F1-score: 0.91")

    st.divider()

    st.write("### Supported Categories")

    for class_name in CLASS_NAMES:
        st.write(f"• {class_name}")

