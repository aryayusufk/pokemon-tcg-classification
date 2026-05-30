import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from streamlit_cropper import st_cropper

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
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


# ==========================================
# 2. INISIALISASI FUNGSI MUAT (CACHE)
# ==========================================
@st.cache_resource
def load_classification_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    model_path = os.path.join(root_dir, "models", "mobilenetv2_pokemon_tcg.keras")
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_booster_reference():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    json_path = os.path.join(root_dir, "data", "booster_reference.json")
    with open(json_path, "r") as file:
        return json.load(file)


# ==========================================
# 3. ANTARMUKA UTAMA
# ==========================================
def main():
    st.title("Sistem Klasifikasi dan Rekomendasi Booster Pack Pokémon TCG")
    st.markdown(
        "Sistem cerdas ini mengimplementasikan algoritma *Transfer Learning* (MobileNetV2) dengan pendekatan *Micro-Region of Interest* (Micro-RoI)."
    )

    try:
        model = load_classification_model()
        booster_ref = load_booster_reference()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat komponen sistem: {e}")
        return

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Unggah Citra Kartu (Format: JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # ---------------------------------------------------------
        # PIPA PRA-PEMROSESAN INTERAKTIF
        # ---------------------------------------------------------
        st.markdown("### ✂️ Sesuaikan Garis Tepi Kartu")
        st.info(
            "Geser kotak biru agar PAS dengan batas luar kartu. **Jika Anda menggunakan gambar dari dataset, pastikan kotak biru diperlebar hingga memenuhi 100% gambar!**"
        )

        cropped_image = st_cropper(
            image, realtime_update=True, box_color="#0026FF", aspect_ratio=(420, 588)
        )

        st.markdown("---")
        col_input, col_output = st.columns(2)

        with col_input:
            st.subheader("Pipa Ekstraksi Model")

            # 1. SINKRONISASI INTERPOLASI KERAS (NEAREST)
            img_resized = cropped_image.resize((420, 588), Image.Resampling.NEAREST)
            st.image(
                img_resized,
                caption="Citra Resolusi Native (420x588)",
                use_container_width=True,
            )

            # 2. VISUALISASI MICRO-ROI (FITUR X-RAY)
            # Menghitung koordinat piksel yang sama persis dengan layer Cropping2D di model
            roi_top = int(588 * 0.85)  # Membuang 85% atas (mulai dari piksel 499)
            roi_right = 420 - int(420 * 0.70)  # Menyisakan 30% kiri (sampai piksel 126)

            # Format pemotongan PIL: (kiri, atas, kanan, bawah)
            img_roi = img_resized.crop((0, roi_top, roi_right, 588))

            st.markdown("**Titik Fokus Model (Micro-RoI):**")
            st.image(
                img_roi,
                caption="WAJIB: Simbol set harus terlihat jelas di dalam kotak ini. Jika kosong/terpotong, sesuaikan cropper di atas!",
                width=180,
            )

            # 3. SINKRONISASI TIPE DATA MATRIKS KERAS (FLOAT32)
            img_array = np.array(img_resized, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_array)

        # ---------------------------------------------------------
        # PREDIKSI DAN VISUALISASI LUARAN
        # ---------------------------------------------------------
        predictions = model.predict(img_preprocessed)[0]
        top_3_indices = np.argsort(predictions)[::-1][:3]

        top_1_class = CLASS_NAMES[top_3_indices[0]]
        top_1_confidence = predictions[top_3_indices[0]] * 100

        with col_output:
            st.subheader("Hasil Klasifikasi")
            st.success(
                f"**Identifikasi Utama:** {top_1_class.replace('-', ' ').title()}"
            )
            st.markdown(
                f"**Tingkat Kepercayaan (Confidence):** {top_1_confidence:.2f}%"
            )

            st.markdown("**Top-3 Prediksi Model:**")
            for i in top_3_indices:
                class_label = CLASS_NAMES[i].replace("-", " ").title()
                confidence_score = predictions[i] * 100
                st.write(f"- {class_label}: {confidence_score:.2f}%")

            st.markdown("---")
            st.subheader("Rekomendasi Booster Pack")

            if top_1_class in booster_ref:
                booster_info = booster_ref[top_1_class][0]
                st.markdown(f"**Nama Produk:** {booster_info['nama']}")
                st.markdown(
                    f"**Tahun Rilis:** {booster_info['rilis']} | **Kode Set:** {booster_info.get('kode', 'N/A')}"
                )

                current_dir = os.path.dirname(os.path.abspath(__file__))
                root_dir = os.path.dirname(current_dir)
                booster_img_path = os.path.join(root_dir, booster_info["gambar"])

                if os.path.exists(booster_img_path):
                    st.image(booster_img_path, width=250, caption=booster_info["nama"])
                else:
                    st.warning(
                        "Aset visual kemasan tidak ditemukan di dalam direktori penyimpanan."
                    )
            else:
                st.error(
                    "Data referensi untuk set kartu ini tidak terdaftar di dalam basis data sistem."
                )


if __name__ == "__main__":
    main()
