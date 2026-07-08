import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from streamlit_cropper import st_cropper

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Pokémon TCG Classifier",
    page_icon="assets/Pokémon_Trading_Card_Game_logo.svg",
    layout="wide",
    initial_sidebar_state="expanded",
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
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "mobilenetv2_701515.keras",
    )
    return tf.keras.models.load_model(model_path)


# ==========================================
# 3. ANTARMUKA UTAMA
# ==========================================
def main():
    try:
        model = load_classification_model()
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan saat memuat model sistem: {e}")
        return

    # --- SIDEBAR (PANEL SAMPING) ---
    with st.sidebar:
        st.markdown("### 📖 Cara Penggunaan")
        st.markdown("""
        1. **Unggah Foto:** Masukkan gambar kartu Pokémon dari set yang didukung.
        2. **Potong (Crop):** Pada layar utama, geser kotak biru hingga pas di garis batas tepi kartu.
        3. **Lihat Hasil:** Sistem akan otomatis menampilkan hasil identifikasi set kartu.
        """)

        st.markdown("---")
        st.markdown("### 📂 Input Data")
        uploaded_file = st.file_uploader(
            "Unggah Citra Kartu (JPG/PNG)", type=["jpg", "jpeg", "png"]
        )

        st.markdown("---")
        st.caption(
            "Sistem ini ditenagai oleh Patch-Based MobileNetV2 dengan tingkat kepercayaan ambang batas 75%."
        )

    # --- MAIN AREA (HALAMAN UTAMA) ---
    col1, col2 = st.columns([1, 5], vertical_alignment="center")

    with col1:
        st.image(
            "assets/Pokémon_Trading_Card_Game_logo.svg",
        )
    
    with col2:
        st.title("Klasifikasi Set Kartu Pokémon TCG")

    if uploaded_file is None:
        st.info(
            "Silakan unggah gambar kartu melalui **Panel Input Data** di sebelah kiri untuk memulai."
        )
        st.markdown(
            "> *Catatan: Sistem mendukung pengenalan untuk 10 Set populer era Sword & Shield hingga Scarlet & Violet.*"
        )
    else:
        image = Image.open(uploaded_file).convert("RGB")
        col_left, col_right = st.columns([4, 5], gap="large")

        # KOLOM KIRI: Interaktif Cropper
        with col_left:
            st.markdown("#### ✂️ Penyesuaian Garis Kartu")
            st.caption("Pastikan kotak biru menutupi area kartu secara penuh.")
            cropped_image = st_cropper(
                image,
                realtime_update=True,
                box_color="#0026FF",
                aspect_ratio=(420, 588),
            )

        # Proses Data di Latar Belakang
        width, height = cropped_image.size
        patch_img = cropped_image.crop(
            (0, int(height * 0.85), int(width * 0.30), height)
        )
        patch_resized = patch_img.resize((224, 224), Image.Resampling.LANCZOS)

        img_array = np.array(patch_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array)

        predictions = model.predict(img_preprocessed)[0]
        top_3_indices = np.argsort(predictions)[::-1][:3]

        top_1_class = CLASS_NAMES[top_3_indices[0]]
        top_1_confidence = predictions[top_3_indices[0]] * 100
        CONFIDENCE_THRESHOLD = 75.0

        # KOLOM KANAN: Hasil
        with col_right:
            st.markdown("#### 📊 Hasil Analisis")

            if top_1_confidence >= CONFIDENCE_THRESHOLD:
                st.success(
                    f"**Identifikasi Set:** {top_1_class.replace('-', ' ').title()} ({top_1_confidence:.2f}%)"
                )
            else:
                st.error("⚠️ Gambar Ditolak: Tingkat Kepercayaan Terlalu Rendah.")
                st.markdown(
                    f"**Skor Maksimal:** {top_1_confidence:.2f}% ({top_1_class.replace('-', ' ').title()})"
                )
                st.warning("""
                **Kemungkinan Penyebab:**
                1. Gambar bukan kartu Pokémon TCG.
                2. Kartu berasal dari set di luar 10 kelas yang didukung.
                3. Simbol set di ujung kiri bawah tidak masuk ke dalam kotak crop.
                """)

            st.markdown("---")
            with st.expander("⚙️ Detail Ekstraksi Teknis (Klik untuk memperluas)"):
                st.markdown("**1. Probabilitas Tertinggi (Top-3):**")
                for i in top_3_indices:
                    st.write(
                        f"- {CLASS_NAMES[i].replace('-', ' ').title()}: {predictions[i] * 100:.2f}%"
                    )

                st.markdown("**2. Visualisasi Micro-RoI (Patch):**")
                st.image(
                    patch_resized,
                    caption="Citra input beresolusi 224x224 yang masuk ke model",
                    width=150,
                )


if __name__ == "__main__":
    main()
