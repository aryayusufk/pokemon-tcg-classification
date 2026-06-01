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

    uploaded_file = st.file_uploader(
        "Unggah Citra Kartu", type=["jpg", "jpeg", "png", "jfif"]
    )

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

        # ---------------------------------------------------------
        # PREDIKSI DAN VISUALISASI LUARAN (DENGAN THRESHOLDING)
        # ---------------------------------------------------------
        predictions = model.predict(img_preprocessed)[0]
        top_3_indices = np.argsort(predictions)[::-1][:3]

        top_1_class = CLASS_NAMES[top_3_indices[0]]
        top_1_confidence = predictions[top_3_indices[0]] * 100

        # AMBANG BATAS KEPERCAYAAN (THRESHOLD)
        # Kartu Pokemon yang valid biasanya memiliki skor > 75% dengan model Patch-Based
        CONFIDENCE_THRESHOLD = 75.0

        with col_output:
            st.subheader("Hasil Klasifikasi")

            # Logika Filter Out-of-Distribution (OOD)
            if top_1_confidence >= CONFIDENCE_THRESHOLD:
                # Jika skor tinggi, tampilkan hasil seperti biasa
                st.success(
                    f"**Identifikasi Utama:** {top_1_class.replace('-', ' ').title()} ({top_1_confidence:.2f}%)"
                )

                st.markdown("**Top-3 Prediksi Model:**")
                for i in top_3_indices:
                    st.write(
                        f"- {CLASS_NAMES[i].replace('-', ' ').title()}: {predictions[i] * 100:.2f}%"
                    )

                st.markdown("---")
                st.subheader("Rekomendasi Booster Pack")

                if top_1_class in booster_ref:
                    info = booster_ref[top_1_class][0]
                    st.markdown(f"""
                                **Nama Produk:** {info['nama']}  
                                **Tahun Rilis:** {info['rilis']} | **Kode Set:** {info.get('kode', 'N/A')}
                                """)

                    img_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        info["gambar"],
                    )
                    if os.path.exists(img_path):
                        st.image(img_path, width=250, caption=info["nama"])
            else:
                # Jika skor di bawah threshold (Misal: gambar meja, wajah, dll)
                st.error(
                    "⚠️ Sistem tidak dapat mengidentifikasi gambar ini dengan tingkat kepercayaan yang meyakinkan."
                )
                st.markdown(f"""
                            **Skor Tertinggi:** {top_1_confidence:.2f}% ({top_1_class.replace('-', ' ').title()})
                            
                            **Kemungkinan Penyebab:**
                            1. Gambar yang diunggah **bukan kartu Pokémon TCG**.
                            2. Kartu berasal dari **set di luar 10 kelas yang didukung**.
                            3. Area *Micro-RoI* (kiri bawah) terlalu blur atau tertutup pantulan cahaya.
                            """)
                st.info(
                    "Sesuai batasan sistem, pastikan Anda hanya mengunggah kartu Pokémon TCG dari 10 set yang terdaftar."
                )


if __name__ == "__main__":
    main()
