import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from streamlit_cropper import st_cropper

# ==========================================
# 1. KONFIGURASI HALAMAN (HARUS DI PALING ATAS)
# ==========================================
st.set_page_config(
    page_title="Pokémon TCG Classifier", 
    page_icon="🃏", 
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_NAMES = [
    "151", "destined-rivals", "evolving-skies", "journey-together",
    "lost-origin", "paldea-evolved", "paldean-fates", "prismatic-evolutions",
    "scarlet-violet-promos", "surging-sparks",
]

# ==========================================
# 2. INISIALISASI FUNGSI MUAT (CACHE)
# ==========================================
@st.cache_resource
def load_classification_model():
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "mobilenetv2_701515.keras", # Menggunakan model terbaik
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

# ==========================================
# 3. ANTARMUKA UTAMA
# ==========================================
def main():
    try:
        model = load_classification_model()
        booster_ref = load_booster_reference()
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan saat memuat sistem: {e}")
        return

    # --- SIDEBAR (PANEL SAMPING) ---
    with st.sidebar:
        st.markdown("### 📖 Cara Penggunaan")
        st.markdown("""
        1. **Unggah Foto:** Masukkan gambar kartu Pokémon dari set yang didukung.
        2. **Potong (Crop):** Pada layar utama, geser kotak biru hingga pas di garis batas tepi kartu.
        3. **Lihat Hasil:** Sistem akan otomatis menampilkan set kartu dan rekomendasi produknya.
        """)
        
        st.markdown("---")
        st.markdown("### 📂 Input Data")
        uploaded_file = st.file_uploader("Unggah Citra Kartu (JPG/PNG)", type=["jpg", "jpeg", "png"])
        
        st.markdown("---")
        st.caption("Sistem ini ditenagai oleh Patch-Based MobileNetV2 dengan tingkat kepercayaan ambang batas 75%.")

    # --- MAIN AREA (HALAMAN UTAMA) ---
    st.title("🃏 Klasifikasi Set Kartu & Rekomendasi Booster Pack")
    
    if uploaded_file is None:
        # Tampilan kosong saat belum ada gambar (Welcoming Screen)
        st.info("👈 Silakan unggah gambar kartu melalui **Panel Input Data** di sebelah kiri untuk memulai.")
        st.markdown("> *Catatan: Sistem mendukung pengenalan untuk 10 Set populer era Sword & Shield hingga Scarlet & Violet.*")
    else:
        # Jika gambar diunggah, bagi layar menjadi 2 kolom (Rasio 4:5 agar seimbang)
        image = Image.open(uploaded_file).convert('RGB')
        col_left, col_right = st.columns([4, 5], gap="large")

        # KOLOM KIRI: Interaktif Cropper
        with col_left:
            st.markdown("#### ✂️ Penyesuaian Garis Kartu")
            st.caption("Pastikan kotak biru menutupi area kartu secara penuh.")
            cropped_image = st_cropper(
                image,
                realtime_update=True,
                box_color='#0026FF',
                aspect_ratio=(420, 588)
            )

        # Proses Data di Latar Belakang
        width, height = cropped_image.size
        patch_img = cropped_image.crop((0, int(height * 0.85), int(width * 0.30), height))
        patch_resized = patch_img.resize((224, 224), Image.Resampling.LANCZOS)
        
        img_array = np.array(patch_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array)

        predictions = model.predict(img_preprocessed)[0]
        top_3_indices = np.argsort(predictions)[::-1][:3]
        
        top_1_class = CLASS_NAMES[top_3_indices[0]]
        top_1_confidence = predictions[top_3_indices[0]] * 100
        CONFIDENCE_THRESHOLD = 75.0 

        # KOLOM KANAN: Hasil dan Rekomendasi
        with col_right:
            st.markdown("#### 📊 Hasil Analisis")
            
            if top_1_confidence >= CONFIDENCE_THRESHOLD:
                # Tampilan Sukses
                st.success(f"**Identifikasi Set:** {top_1_class.replace('-', ' ').title()} ({top_1_confidence:.2f}%)")
                
                if top_1_class in booster_ref:
                    info = booster_ref[top_1_class][0]
                    
                    # Kotak Informasi Rekomendasi
                    st.markdown("#### 📦 Rekomendasi Produk")
                    st.markdown(f"**{info['nama']}**")
                    st.caption(f"Tahun Rilis: {info['rilis']} | Kode Set: {info.get('kode', 'N/A')}")
                    
                    # Memanggil gambar booster
                    img_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        info["gambar"]
                    )
                    if os.path.exists(img_path):
                        # Menampilkan gambar booster dengan ukuran terkontrol agar tidak makan tempat
                        st.image(img_path, width=220)
                
            else:
                # Tampilan Penolakan (Defensive UI)
                st.error("⚠️ Gambar Ditolak: Tingkat Kepercayaan Terlalu Rendah.")
                st.markdown(f"**Skor Maksimal:** {top_1_confidence:.2f}% ({top_1_class.replace('-', ' ').title()})")
                st.warning("""
                **Kemungkinan Penyebab:**
                1. Gambar bukan kartu Pokémon TCG.
                2. Kartu berasal dari set di luar 10 kelas yang didukung.
                3. Simbol set di ujung kiri bawah tidak masuk ke dalam kotak crop.
                """)

            # Fitur Expander: Menyembunyikan data teknis (X-Ray) agar UI tetap bersih
            st.markdown("---")
            with st.expander("⚙️ Detail Ekstraksi Teknis (Klik untuk memperluas)"):
                st.markdown("**1. Probabilitas Tertinggi (Top-3):**")
                for i in top_3_indices:
                    st.write(f"- {CLASS_NAMES[i].replace('-', ' ').title()}: {predictions[i] * 100:.2f}%")
                
                st.markdown("**2. Visualisasi Micro-RoI (Patch):**")
                st.image(patch_resized, caption="Citra input beresolusi 224x224 yang masuk ke model", width=150)

if __name__ == "__main__":
    main()