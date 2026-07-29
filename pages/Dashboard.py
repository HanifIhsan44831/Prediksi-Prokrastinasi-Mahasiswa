import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================
@st.cache_data
def load_data():
    return pd.read_csv("datasetfix.csv")

df = load_data()

# Bersihkan nama kolom
df.columns = (
    df.columns
    .str.strip()
    .str.replace("\n", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
)

# ==========================
# DASHBOARD
# ==========================

st.title("📊 Dashboard")

st.markdown("""
Selamat datang pada **Sistem Prediksi Tingkat Prokrastinasi Mahasiswa**
menggunakan algoritma **Categorical Naïve Bayes**.
""")

st.divider()

# ==========================
# METRIC
# ==========================

jumlah_data = len(df)
jumlah_variabel = len(df.columns)
jumlah_kelas = df["Target"].nunique()
akurasi = 35

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📄 Jumlah Data", jumlah_data)

with col2:
    st.metric("📚 Variabel", jumlah_variabel)

with col3:
    st.metric("🎯 Kategori", jumlah_kelas)

with col4:
    st.metric("🤖 Akurasi", f"{akurasi}%")

st.divider()

# ==========================
# DISTRIBUSI TARGET
# ==========================

st.subheader("Distribusi Tingkat Prokrastinasi")

target_count = (
    df["Target"]
    .value_counts()
    .reset_index()
)

target_count.columns = ["Kategori", "Jumlah"]

fig = px.bar(
    target_count,
    x="Kategori",
    y="Jumlah",
    color="Kategori",
    text="Jumlah"
)

fig.update_layout(height=450)

st.plotly_chart(fig, use_container_width=True)

st.info("""
Grafik di atas menunjukkan distribusi tingkat prokrastinasi mahasiswa.
Kategori **Sedang** merupakan kategori yang paling banyak ditemukan pada dataset penelitian.
""")

st.divider()

# ==========================
# RINGKASAN PENELITIAN
# ==========================





# ==========================
# PREVIEW DATASET
# ==========================

st.subheader("Preview Dataset")

st.dataframe(df.head(10), use_container_width=True)