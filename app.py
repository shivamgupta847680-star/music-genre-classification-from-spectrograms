
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵"
)

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("music_genre_model.keras")

st.title("🎵 Music Genre Classification")
st.write("Upload a spectrogram image to predict its music genre.")

try:
    model = load_model()

    uploaded_file = st.file_uploader(
        "Choose a spectrogram image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded spectrogram")

        if st.button("Predict Genre"):
            image = image.resize((128, 128))
            arr = np.array(image, dtype=np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)

            predictions = model.predict(arr, verbose=0)[0]
            index = int(np.argmax(predictions))

            st.success(
                f"Predicted Genre: {GENRES[index].title()}"
            )
            st.write(
                f"Model score: {predictions[index] * 100:.2f}%"
            )

except Exception as e:
    st.error(f"Error: {e}")
