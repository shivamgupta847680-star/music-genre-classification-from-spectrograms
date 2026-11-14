
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
def load_models():
    genre_model = tf.keras.models.load_model(
        "music_genre_model.keras"
    )
    validator_model = tf.keras.models.load_model(
        "spectrogram_validator.keras"
    )
    return genre_model, validator_model

st.title("🎵 Music Genre Classification")
st.write("Upload a spectrogram image to predict its music genre.")

try:
    genre_model, validator_model = load_models()

    uploaded_file = st.file_uploader(
        "Choose a spectrogram image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")

        if st.button("Predict Genre"):
            resized_image = image.resize((128, 128))

            # Validator input: raw pixel values (0-255)
            raw_arr = np.array(
                resized_image, dtype=np.float32
            )
            validator_input = np.expand_dims(raw_arr, axis=0)

            validation_score = validator_model.predict(
                validator_input, verbose=0
            )[0][0]

            if validation_score < 0.5:
                st.error(
                    "Invalid Image! Please upload a spectrogram image."
                )
            else:
                # Genre model input: normalized pixel values (0-1)
                genre_arr = np.expand_dims(
                    raw_arr / 255.0, axis=0
                )

                predictions = genre_model.predict(
                    genre_arr, verbose=0
                )[0]

                index = int(np.argmax(predictions))

                st.success(
                    f"Predicted Genre: {GENRES[index].title()}"
                )
                st.write(
                    f"Genre Model Score: "
                    f"{predictions[index] * 100:.2f}%"
                )

except Exception as e:
    st.error(f"Error: {e}")
