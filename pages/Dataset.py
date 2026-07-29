import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Dataset",
    page_icon="📁",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================
@st.cache_data
def load_data():
    return pd.read_csv("datasetfix.csv")

df = load_data()

# Membersihkan nama kolom
df.columns = (
    df.columns
    .str.strip()
    .str.replace("\n", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
)

# ==========================
# HEADER
# ==========================

st.title("📁 Dataset Penelitian")

st.markdown("""
Halaman ini menampilkan dataset penelitian yang digunakan
untuk membangun model prediksi menggunakan
algoritma **Categorical Naïve Bayes**.
""")

st.divider()

# ==========================
# METRIC
# ==========================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📄 Jumlah Data", len(df))

with col2:
    st.metric("📚 Jumlah Variabel", len(df.columns))

with col3:
    st.metric("🎯 Jumlah Kelas", df["Target"].nunique())

st.divider()

# ==========================
# SEARCH
# ==========================

st.subheader("🔍 Pencarian Data")

keyword = st.text_input(
    "Cari berdasarkan Nama, NIM, atau Angkatan"
)

df_show = df.copy()

if keyword:
    keyword = keyword.lower()

    mask = df_show.astype(str).apply(
        lambda x: x.str.lower().str.contains(keyword)
    ).any(axis=1)

    df_show = df_show[mask]

# ==========================
# DOWNLOAD
# ==========================

st.download_button(
    label="📥 Download Dataset CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="dataset_penelitian.csv",
    mime="text/csv"
)

st.divider()

# ==========================
# DATAFRAME
# ==========================

st.subheader("📋 Dataset")

st.dataframe(
    df_show,
    use_container_width=True,
    height=500
)

st.caption(f"Menampilkan {len(df_show)} dari {len(df)} data.")

st.divider()

# ==========================
# STATISTIK
# ==========================

st.subheader("📊 Ringkasan Dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Jumlah Responden", len(df))

with col2:
    st.metric("Jumlah Variabel", len(df.columns))

with col3:
    st.metric("Kategori Target", df["Target"].nunique())

with col4:
    st.metric("Data Lengkap", f"{df.notna().all(axis=1).sum()} Baris")

st.divider()

# ==========================
# INFORMASI
# ==========================

st.info("""
Dataset penelitian terdiri dari **300 responden**
mahasiswa Teknik Informatika UMMI.

Dataset ini memuat variabel:

- Angkatan
- Jenis Kelamin
- Durasi Penggunaan Media Sosial
- Frekuensi Membuka Media Sosial
- Waktu Penggunaan
- Platform Media Sosial
- Kebiasaan Mengecek Notifikasi

Target klasifikasi terdiri dari:

- Rendah
- Sedang
- Tinggi
""")