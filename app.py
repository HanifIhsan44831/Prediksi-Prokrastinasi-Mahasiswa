import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Prokrastinasi Mahasiswa", page_icon="🎓", layout="wide")

@st.cache_resource
def load_artifact():
    return joblib.load("model.pkl")

artifact=load_artifact()
model=artifact["model"]
encoder=artifact["encoder"]
features=artifact["features"]

OPTIONS={'Angkatan': ['2020', '2021', '2022', '2023', '2024', '2025'], 'Jenis Kelamin': ['Laki - Laki', 'Perempuan'], '  Berapa rata-rata durasi Anda menggunakan media sosial dalam sehari?  ': ['a. < 1 jam', 'b. 1-3 jam', 'c. 3 - 5 Jam', 'd. > 5 Jam'], 'Seberapa sering Anda membuka media sosial dalam satu hari? ': ['a. (< 5 kali)', 'b. (5 - 10 kali)', 'c. (11 - 20 kali)', 'd. (> 20 kali / Hampir setiap saat)'], 'Pada waktu kapan Anda paling sering mengakses media sosial? ': ['a. Pagi hari', 'b. Siang hari (Istirahat kuliah)', 'c. Malam hari (Sebelum tidur)', 'd. Tengah malam (Begadang)'], 'Platform media sosial apa yang paling sering Anda gunakan? ': ['a. Instagram', 'b. TikTok', 'c. Twitter (X)', 'd. YouTube', 'e. WhatsApp'], 'Apakah Anda sering mengecek notifikasi media sosial saat sedang belajar/mengerjakan tugas?  ': ['a. Tidak Pernah', 'b. Jarang', 'c. Sering', 'd. Selalu']}

st.title("🎓 Prediksi Tingkat Prokrastinasi Mahasiswa")
st.caption("Categorical Naïve Bayes • Pola Penggunaan Media Sosial")
st.info("Masukkan karakteristik mahasiswa untuk melakukan testing terhadap model hasil penelitian.")

labels = [
    "Angkatan", "Jenis Kelamin", "Durasi penggunaan media sosial per hari",
    "Frekuensi membuka media sosial", "Waktu paling sering mengakses media sosial",
    "Platform yang paling sering digunakan", "Kebiasaan mengecek notifikasi saat belajar"
]

values=[]
with st.form("prediksi"):
    c1,c2=st.columns(2)
    for i,(col,label) in enumerate(zip(features,labels)):
        box=c1 if i%2==0 else c2
        with box:
            values.append(st.selectbox(label, OPTIONS[col], key=str(i)))
    submit=st.form_submit_button("🔍 Prediksi Sekarang", use_container_width=True)

if submit:
    input_df=pd.DataFrame([values],columns=features)
    encoded=encoder.transform(input_df)+1
    pred=model.predict(encoded)[0]
    prob=model.predict_proba(encoded)[0]

    st.divider()
    st.subheader("Hasil Prediksi")
    if pred=="Rendah":
        st.success(f"TINGKAT PROKRASTINASI: {pred.upper()}")
    elif pred=="Sedang":
        st.warning(f"TINGKAT PROKRASTINASI: {pred.upper()}")
    else:
        st.error(f"TINGKAT PROKRASTINASI: {pred.upper()}")

    st.write("Probabilitas model untuk setiap kategori:")
    cols=st.columns(len(model.classes_))
    for box,cls,p in zip(cols,model.classes_,prob):
        box.metric(cls, f"{p*100:.2f}%")

    st.caption("Hasil merupakan prediksi model machine learning dan bukan diagnosis psikologis.")

with st.expander("Tentang Model"):
    st.write("""
    Aplikasi ini menggunakan algoritma Categorical Naïve Bayes untuk memprediksi
    tingkat prokrastinasi akademik berdasarkan tujuh variabel prediktor penelitian.
    Model dilatih dari dataset penelitian dan hasil input pengguna diproses dengan
    encoder yang sama seperti data pelatihan.
    """)
