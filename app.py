# app.py
import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
try:
    model = joblib.load("model_tuned.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"❌ Gagal memuat model atau scaler: {e}")
    st.stop()

st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("🔍 Aplikasi Prediksi Tingkat Obesitas")
st.markdown("Masukkan informasi berikut untuk memprediksi tingkat obesitas Anda:")

# Input numerik
age = st.slider("Umur", 10, 100, 25)
height = st.number_input("Tinggi Badan (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
weight = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.5)
fcvc = st.slider("Frekuensi Konsumsi Sayur (1–3)", 1.0, 3.0, 2.0)
ncp = st.slider("Jumlah Makan Utama per Hari", 1.0, 4.0, 3.0)
ch2o = st.slider("Konsumsi Air Harian (liter)", 1.0, 3.0, 2.0)
faf = st.slider("Frekuensi Aktivitas Fisik Mingguan", 0.0, 3.0, 1.0)
tue = st.slider("Waktu Layar Harian (jam)", 0.0, 3.0, 1.0)

# Input kategorikal
favc = st.selectbox("Sering Konsumsi Makanan Tinggi Kalori?", ["yes", "no"])
smoke = st.selectbox("Merokok?", ["yes", "no"])
scc = st.selectbox("Pemantauan Kalori?", ["yes", "no"])
family_history = st.selectbox("Riwayat Keluarga Kegemukan?", ["yes", "no"])
caec = st.selectbox("Kebiasaan Camilan", ["no", "Sometimes", "Frequently", "Always"])
calc = st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Transportasi Utama", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

# Encoding kategorikal
binary_map = {"yes": 1, "no": 0}
caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
mtrans_map = {
    "Walking": 0,
    "Bike": 1,
    "Motorbike": 2,
    "Automobile": 3,
    "Public_Transportation": 4
}

# Susun data input
input_data = np.array([[
    age, height, weight, fcvc, ncp, ch2o, faf, tue,
    binary_map[favc], binary_map[smoke], binary_map[scc], binary_map[family_history],
    caec_map[caec], calc_map[calc], mtrans_map[mtrans]
]])

# Lakukan scaling
try:
    scaled_data = scaler.transform(input_data)
except Exception as e:
    st.error(f"❌ Gagal melakukan scaling data: {e}")
    st.stop()

# Prediksi saat tombol diklik
if st.button("Prediksi"):
    try:
        pred = model.predict(scaled_data)[0]
        st.success(f"Hasil Prediksi: **{pred}**")
    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi: {e}")
