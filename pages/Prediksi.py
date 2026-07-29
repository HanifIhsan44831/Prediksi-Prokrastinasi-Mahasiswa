import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Prediksi",
    page_icon="🤖",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================

@st.cache_resource
def load_artifact():
    return joblib.load("model.pkl")

artifact = load_artifact()

model = artifact["model"]
encoder = artifact["encoder"]
features = artifact["features"]

st.title("🤖 Prediksi Tingkat Prokrastinasi")

st.write("Silakan isi data mahasiswa di bawah ini.")

st.divider()
angkatan = st.selectbox(
    "Angkatan",
    [
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025"
    ]
)

jk = st.selectbox(
    "Jenis Kelamin",
    [
        "Laki - Laki",
        "Perempuan"
    ]
)

durasi = st.selectbox(
    "Durasi Penggunaan Media Sosial",
    [
     "a. < 1 jam",
     "b. 1-3 jam",
     "c. 3 - 5 Jam",
     "d. > 5 Jam"
    ]
)

frekuensi = st.selectbox(
    "Frekuensi Membuka Media Sosial",
    [
        "a. (< 5 kali)",
        "b. (5 - 10 kali)",
        "c. (11 - 20 kali)",
        "d. (> 20 kali / Hampir setiap saat)"
    ]
)

waktu = st.selectbox(
    "Waktu Penggunaan",
    [
        "a. Pagi hari",
        "b. Siang hari (Istirahat kuliah)",
        "c. Malam hari (Sebelum tidur)",
        "d. Tengah malam (Begadang)"
    ]
)

platform = st.selectbox(
    "Platform",
    [
        "a. Instagram",
        "b. TikTok",
        "c. Twitter (X)",
        "d. YouTube",
        "e. WhatsApp"
    ]
)

notifikasi = st.selectbox(
    "Mengecek Notifikasi",
    [
     "a. Tidak Pernah",
     "b. Jarang",
     "c. Sering",
     "d. Selalu"
    ]
)
st.divider()

if st.button("🔍 Prediksi", use_container_width=True):

    # Data harus sesuai urutan features pada model
    input_df = pd.DataFrame([[
        angkatan,
        jk,
        durasi,
        frekuensi,
        waktu,
        platform,
        notifikasi
    ]], columns=features)

    # Encode
    st.write(input_df)
    input_encoded = encoder.transform(input_df)
    
    # Prediksi
    hasil = model.predict(input_encoded)[0]

    # Probabilitas
    probabilitas = model.predict_proba(input_encoded)[0]

    st.success(f"### Hasil Prediksi : **{hasil}**")

    st.subheader("Probabilitas Tiap Kelas")

    prob_df = pd.DataFrame({
        "Kategori": model.classes_,
        "Probabilitas": probabilitas
    })

    st.bar_chart(
        prob_df.set_index("Kategori")
    )

    st.dataframe(prob_df, use_container_width=True)