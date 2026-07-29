import streamlit as st

st.set_page_config(
    page_title="Prediksi Prokrastinasi Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 Sistem Prediksi Tingkat Prokrastinasi Mahasiswa")

st.markdown("""
Selamat datang di aplikasi prediksi tingkat prokrastinasi akademik mahasiswa
menggunakan algoritma **Categorical Naïve Bayes**.

### 📌 Fitur Aplikasi
- 📊 Dashboard
- 📁 Dataset
- 📈 Analisis Data
- 🤖 Prediksi Prokrastinasi
- ℹ️ Tentang Penelitian

Silakan pilih menu pada **sidebar** di sebelah kiri untuk mulai menggunakan aplikasi.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.info("""
### 🎯 Tujuan Penelitian
Memprediksi tingkat prokrastinasi mahasiswa berdasarkan pola penggunaan media sosial
menggunakan algoritma Categorical Naïve Bayes.
""")

with col2:
    st.success("""
### 📚 Informasi Singkat
- **Metode** : CRISP-DM
- **Algoritma** : Categorical Naïve Bayes
- **Dataset** : 300 Responden
- **Kelas** : Rendah, Sedang, Tinggi
""")

st.divider()

st.caption("© 2026 Sistem Prediksi Prokrastinasi Mahasiswa | Teknik Informatika UMMI")