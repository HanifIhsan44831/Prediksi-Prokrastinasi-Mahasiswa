import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Analisis Dataset",
    page_icon="📈",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================
@st.cache_data
def load_data():
    return pd.read_csv("datasetfix.csv")

df = load_data()


st.title("📈 Analisis Dataset")

st.markdown("""
Halaman ini menampilkan analisis deskriptif terhadap dataset penelitian
yang digunakan dalam proses pembangunan model **Categorical Naïve Bayes**.
""")

st.divider()

# ==========================
# DISTRIBUSI TARGET
# ==========================

st.subheader("1. Distribusi Tingkat Prokrastinasi")

target = "Target"

target_count = df[target].value_counts().reset_index()
target_count.columns = [target, "Jumlah"]

fig = px.bar(
    target_count,
    x=target,
    y="Jumlah",
    text="Jumlah",
    color=target
)

fig.update_layout(height=450)

st.plotly_chart(fig, use_container_width=True)

st.info("""
Grafik di atas menunjukkan jumlah mahasiswa pada masing-masing kategori
prokrastinasi akademik.

Distribusi kelas ini penting untuk mengetahui apakah dataset seimbang
atau terdapat kelas yang mendominasi sehingga dapat mempengaruhi
kinerja model klasifikasi.
""")

st.divider()

# ==========================
# PLATFORM
# ==========================

st.subheader("2. Platform Media Sosial")

platform = "Platform media sosial apa yang paling sering Anda gunakan? "

platform_count = df[platform].value_counts().reset_index()
platform_count.columns = [platform, "Jumlah"]

fig2 = px.pie(
    platform_count,
    names=platform,
    values="Jumlah",
    hole=0.45
)

fig2.update_layout(height=500)

st.plotly_chart(fig2, use_container_width=True)

st.info("""
Diagram lingkaran menunjukkan platform media sosial yang paling sering
digunakan oleh responden penelitian.
""")

st.divider()

# ==========================
# DURASI
# ==========================

st.subheader("3. Durasi Penggunaan Media Sosial")

# Cari kolom durasi secara otomatis
durasi = next(col for col in df.columns if "durasi" in col.lower())

durasi_count = (
    df[durasi]
    .value_counts()
    .reset_index()
)

durasi_count.columns = ["Durasi", "Jumlah"]

fig3 = px.bar(
    durasi_count,
    x="Durasi",
    y="Jumlah",
    text="Jumlah",
    color="Durasi"
)

fig3.update_layout(height=450)

st.plotly_chart(fig3, use_container_width=True)

st.info("""
Grafik ini memperlihatkan distribusi lama penggunaan media sosial
setiap hari oleh responden.
""")

st.divider()

# ==========================
# STATISTIK
# ==========================

st.subheader("4. Statistik Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Jumlah Data", len(df))

with col2:
    st.metric("Jumlah Variabel", len(df.columns))

with col3:
    st.metric("Jumlah Kelas", df[target].nunique())

