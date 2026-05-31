import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from streamlit_cropper import st_cropper

st.set_page_config(
    page_title="Klasifikasi Set Kartu Pokémon", page_icon="🃏", layout="wide"
)

CLASS_NAMES = [
    "151",
    "destined-rivals",
    "evolving-skies",
    "journey-together",
    "lost-origin",
    "paldea-evolved",
    "paldean-fates",
    "prismatic-evolutions",
    "scarlet-violet-promos",
    "surging-sparks",
]


@st.cache_resource
def load_classification_model():
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "mobilenetv2_pokemon_tcg.keras",
    )
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_booster_reference():
    json_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "booster_reference.json",
    )
    with open(json_path, "r") as file:
        return json.load(file)


def main():
    st.title("Sistem Klasifikasi Patch-Based Pokémon TCG")

    try:
        model = load_classification_model()
        booster_ref = load_booster_reference()
    except Exception as e:
        st.error(f"Error: {e}")
        return

    uploaded_file = st.file_uploader("Unggah Citra Kartu", type=["jpg", "jpeg", "png", "jfif"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("### ✂️ Sesuaikan Garis Tepi Kartu")
        cropped_image = st_cropper(
            image, realtime_update=True, box_color="#0026FF", aspect_ratio=(420, 588)
        )

        col_input, col_output = st.columns(2)

        with col_input:
            st.subheader("Patch-Based Extraction")

            # 1. Dapatkan ukuran gambar hasil crop interaktif
            width, height = cropped_image.size

            # 2. Ekstraksi koordinat Micro-RoI persis seperti saat training
            left = 0
            top = int(height * 0.85)
            right = int(width * 0.30)
            bottom = height

            # 3. Potong dan Zoom-in
            patch_img = cropped_image.crop((left, top, right, bottom))
            patch_resized = patch_img.resize((224, 224), Image.Resampling.LANCZOS)

            st.image(patch_resized, caption="Patch Input ke Model (224x224)", width=224)

            # 4. Prediksi
            img_array = np.array(patch_resized, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_array)

        predictions = model.predict(img_preprocessed)[0]
        top_3_indices = np.argsort(predictions)[::-1][:3]
        top_1_class = CLASS_NAMES[top_3_indices[0]]

        with col_output:
            st.subheader("Hasil Klasifikasi")
            st.success(
                f"**Identifikasi Utama:** {top_1_class.replace('-', ' ').title()} ({predictions[top_3_indices[0]] * 100:.2f}%)"
            )
            for i in top_3_indices:
                st.write(f"- {CLASS_NAMES[i].title()}: {predictions[i] * 100:.2f}%")

            if top_1_class in booster_ref:
                info = booster_ref[top_1_class][0]
                img_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    info["gambar"],
                )
                if os.path.exists(img_path):
                    st.image(img_path, width=250, caption=info["nama"])


if __name__ == "__main__":
    main()
