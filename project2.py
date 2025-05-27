import numpy as np
import pandas as pd
import streamlit as st
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Data
data = {
    "Kategori": ["Nongkrong"] * 10 + ["Nugas"] * 5 + ["Restoran"] * 6 + ["Ramah Kantong"] * 7,
    "Nama Tempat": [
    "Perwira Kopi", "RP Coffee", "Warmindo Well", "Kaze HQ", "Sadane Coffee", "Vion Coffee", "Yugata Coffee", "Warkop Pancong Lumer", "Talk Coffee Industri", "Kisah Lama Kopi",
    "Kaze HQ", "Yarra", "Its Brewme", "Ammo Coffee", "Hallo Burjois",
    "Umm A Yummy", "Cafe Loco", "Pasta Kangen Karawang", "Wardes", "Bebek Kaleyo Galuh Mas", "Riz Cafe dan Resto",
    "Sego Pedes Basman", "Soto Segerrr & Frozen", "Bubur & Soto Ayam Depan UNSIKA", "Mie Mukidi",
    "Warteg Putra", "Warteg Risky", "Warteg Kebumen"
    ],
    "Harga": [16547, 19350, 6756, 26283, 16966, 33120, 22037, 7711, 25549, 15684,
              26283, 25934, 15297, 20000, 22868, 13245, 12666,
              16510, 57255, 13798, 26247, 
              6549, 8363, 5312, 9116, 6567, 5234, 5786],
    "Waktu": [1, 4, 1, 1, 6, 2, 4, 1, 8, 8,
              1, 3, 4, 1, 4,
              1, 1, 3, 3, 10, 10,
              1, 1, 1, 2, 1, 1, 1],
    "Suasana": [3, 3, 3, 5, 4, 4, 4, 1, 4, 2,
                5, 4, 5, 3, 5,
                3, 3, 2, 3, 5, 4,
                2, 3, 3, 2, 2, 2, 2],
    "Kenyamanan": [3, 4, 3, 5, 4, 4, 4, 2, 4, 3,
                   5, 5, 5, 3, 5,
                   4, 3, 2, 3, 5, 4,
                   2, 2, 3, 2, 3, 3, 3],
    "Fasilitas": [4, 3, 3, 5, 3, 4, 3, 3, 4, 3,
                  5, 5, 4, 3, 4,
                  4, 2, 2, 4, 5, 5,
                  2, 2, 2, 2, 3, 3, 3]
}
df = pd.DataFrame(data)

# Fuzzy Logic Variables
harga = ctrl.Antecedent(np.arange(5000, 60001, 1000), 'harga')
waktu = ctrl.Antecedent(np.arange(1, 11, 1), 'waktu')
suasana = ctrl.Antecedent(np.arange(1, 6, 1), 'suasana')
kenyamanan = ctrl.Antecedent(np.arange(1, 6, 1), 'kenyamanan')
fasilitas = ctrl.Antecedent(np.arange(1, 6, 1), 'fasilitas')
rekomendasi = ctrl.Consequent(np.arange(0, 101, 1), 'rekomendasi')

# Membership Functions
harga['murah'] = fuzz.trimf(harga.universe, [5000, 5000, 15000])
harga['sedang'] = fuzz.trimf(harga.universe, [15000, 22500, 30000])
harga['mahal'] = fuzz.trimf(harga.universe, [30000, 60000, 60000])

waktu['cepat'] = fuzz.trimf(waktu.universe, [1, 1, 4])
waktu['sedang'] = fuzz.trimf(waktu.universe, [2, 5, 8])
waktu['lama'] = fuzz.trimf(waktu.universe, [6, 10, 10])

for var in [suasana, kenyamanan, fasilitas]:
    var['buruk'] = fuzz.trimf(var.universe, [1, 1, 3])
    var['sedang'] = fuzz.trimf(var.universe, [2, 3, 4])
    var['bagus'] = fuzz.trimf(var.universe, [3, 5, 5])

rekomendasi['rendah'] = fuzz.trimf(rekomendasi.universe, [0, 0, 50])
rekomendasi['sedang'] = fuzz.trimf(rekomendasi.universe, [30, 50, 70])
rekomendasi['tinggi'] = fuzz.trimf(rekomendasi.universe, [60, 100, 100])

# Rules
rules = [
    ctrl.Rule(harga['murah'] & waktu['cepat'] & fasilitas['bagus'], rekomendasi['tinggi']),
    ctrl.Rule(harga['murah'] & kenyamanan['bagus'], rekomendasi['tinggi']),
    ctrl.Rule(harga['sedang'] & fasilitas['sedang'], rekomendasi['sedang']),
    ctrl.Rule(harga['mahal'] & waktu['lama'], rekomendasi['rendah']),
    ctrl.Rule(suasana['buruk'] | fasilitas['buruk'], rekomendasi['rendah']),
    ctrl.Rule(suasana['bagus'] & kenyamanan['bagus'], rekomendasi['tinggi']),
    ctrl.Rule(harga['mahal'] & suasana['bagus'] & kenyamanan['bagus'], rekomendasi['sedang']),
    ctrl.Rule(harga['murah'] & suasana['buruk'], rekomendasi['rendah']),
    ctrl.Rule(harga['sedang'] & waktu['cepat'] & fasilitas['bagus'], rekomendasi['tinggi']),
    ctrl.Rule(harga['sedang'] & waktu['sedang'] & suasana['sedang'] & kenyamanan['sedang'] & fasilitas['sedang'], rekomendasi['sedang']),
    ctrl.Rule(fasilitas['bagus'] & kenyamanan['bagus'], rekomendasi['tinggi']),
    ctrl.Rule(suasana['buruk'] & kenyamanan['buruk'], rekomendasi['rendah']),
    ctrl.Rule(waktu['cepat'] & kenyamanan['sedang'] & fasilitas['sedang'], rekomendasi['sedang']),
    ctrl.Rule(waktu['cepat'] & suasana['bagus'] & fasilitas['bagus'], rekomendasi['tinggi']),
    ctrl.Rule(harga['murah'] & fasilitas['sedang'], rekomendasi['sedang']),
]

# Control System
rekom_ctrl = ctrl.ControlSystem(rules)

# Function to calculate fuzzy score
def hitung_skor_fuzzy(row):
    sim = ctrl.ControlSystemSimulation(rekom_ctrl)
    try:
        sim.input['harga'] = row['Harga']
        sim.input['waktu'] = row['Waktu']
        sim.input['suasana'] = row['Suasana']
        sim.input['kenyamanan'] = row['Kenyamanan']
        sim.input['fasilitas'] = row['Fasilitas']
        sim.compute()
        return round(sim.output['rekomendasi'], 2)
    except:
        return 0

# Streamlit page setup
st.set_page_config(page_title="Rekomendasi Tempat", layout="centered")
st.title(" Sistem Rekomendasi Tempat Berbasis Fuzzy")

# User Input for Category
kategori = st.selectbox("Pilih kategori tempat:", sorted(df["Kategori"].unique()))
df_filtered = df[df["Kategori"] == kategori].copy()

# If filtered data is available, process it
if not df_filtered.empty:
    with st.spinner("Menghitung skor fuzzy..."):
        df_filtered["Skor Fuzzy"] = df_filtered.apply(hitung_skor_fuzzy, axis=1)
        df_sorted = df_filtered.sort_values(by="Skor Fuzzy", ascending=False)

    # Show the sorted results
    st.subheader(f"Hasil Rekomendasi untuk Kategori: {kategori}")
    st.dataframe(df_sorted.reset_index(drop=True)[[
        "Nama Tempat", "Harga", "Waktu", "Suasana", "Kenyamanan", "Fasilitas", "Skor Fuzzy"
    ]])
else:
    st.warning("Kategori tidak ditemukan dalam data.")

# Penjelasan indikator
indikator_data = {
    "Indikator": [
        "Harga (Rp)", "Harga (Rp)", "Harga (Rp)",
        "Waktu Pelayanan", "Waktu Pelayanan", "Waktu Pelayanan",
        "Suasana", "Suasana", "Suasana",
        "Kenyamanan", "Kenyamanan", "Kenyamanan",
        "Fasilitas", "Fasilitas", "Fasilitas",
        "Skor Fuzzy", "Skor Fuzzy", "Skor Fuzzy"
    ],
    "Kriteria": [
        "Murah", "Sedang", "Mahal",
        "Dekat", "Sedang", "Jauh",
        "Buruk", "Sedang", "Bagus",
        "Buruk", "Sedang", "Bagus",
        "Buruk", "Sedang", "Bagus",
        "Rendah", "Sedang", "Tinggi"
    ],
    "Rentang Nilai": [
        "< 15.000", "15.000 - 30.000", "> 30.000",
        "1 - 4", "2 - 8", "6 - 10",
        "1 - 3", "2 - 4", "3 - 5",
        "1 - 3", "2 - 4", "3 - 5",
        "1 - 3", "2 - 4", "3 - 5",
        "0 - 50", "30 - 70", "60 - 100"
    ]
}

# Show the table with explanation of fuzzy indicators
tabel_indikator = pd.DataFrame(indikator_data)
with st.expander("Indikator Penilaian Fuzzy"):
    st.table(tabel_indikator)

# User preferences form
st.subheader("Cari Berdasarkan Preferensi Anda")
with st.form("form_preferensi"):
    input_harga = st.slider("Harga yang Diinginkan (Rp)", min_value=5000, max_value=60000, step=1000, value=20000)
    input_waktu = st.slider("Waktu Tempuh (1: Dekat, 10: Jauh)", 1, 10, 4)
    input_suasana = st.slider("Suasana (1: Buruk, 5: Bagus)", 1, 5, 3)
    input_kenyamanan = st.slider("Kenyamanan (1: Buruk, 5: Bagus)", 1, 5, 3)
    input_fasilitas = st.slider("Fasilitas (1: Buruk, 5: Bagus)", 1, 5, 3)
    
    submit = st.form_submit_button("Cari Rekomendasi")

if submit:
    sim_user = ctrl.ControlSystemSimulation(rekom_ctrl)
    sim_user.input['harga'] = input_harga
    sim_user.input['waktu'] = input_waktu
    sim_user.input['suasana'] = input_suasana
    sim_user.input['kenyamanan'] = input_kenyamanan
    sim_user.input['fasilitas'] = input_fasilitas
    sim_user.compute()
    skor_user = sim_user.output['rekomendasi']
    
    st.info(f"Skor Fuzzy Berdasarkan Preferensi Anda: **{round(skor_user, 2)}**")

    # Filtering the places based on user input and fuzzy score
    df["Skor Fuzzy"] = df.apply(hitung_skor_fuzzy, axis=1)
    hasil = df.copy()
    hasil["Selisih Skor"] = abs(hasil["Skor Fuzzy"] - skor_user)
    
    # Filter by waktu input constraint
    hasil_filtered = hasil[hasil['Waktu'] <= input_waktu]
    
    # Sort by the closest score
    hasil_sorted = hasil_filtered.sort_values("Selisih Skor").reset_index(drop=True)
    hasil_sorted = hasil_sorted[hasil_sorted["Selisih Skor"] <= 15]  # Toleransi selisih skor

    if not hasil_sorted.empty:
        st.success("Rekomendasi Tempat yang Cocok dengan Preferensi Anda:")
        st.dataframe(hasil_sorted[[ 
            "Nama Tempat", "Kategori", "Harga", "Waktu", "Suasana", "Kenyamanan", "Fasilitas", "Skor Fuzzy"
        ]])
    else:
        st.warning("Tidak ada tempat yang cocok dengan preferensi Anda.")
