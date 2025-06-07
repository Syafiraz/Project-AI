import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# ------------------ DATA ------------------ #
data = {
    "Kategori": [
        "Nongkrong", "Nongkrong", "Nongkrong", "Nongkrong", "Nongkrong",
        "Nongkrong", "Nongkrong", "Nongkrong", "Nongkrong", "Nongkrong",
        "Nugas", "Nugas", "Nugas", "Nugas", "Nugas",
        "Restoran", "Restoran", "Restoran", "Restoran", "Restoran", "Restoran",
        "Ramah Kantong", "Ramah Kantong", "Ramah Kantong", "Ramah Kantong",
        "Ramah Kantong", "Ramah Kantong", "Ramah Kantong"
    ],
    "Nama Tempat": [
        "Perwira Kopi", "RP Coffee", "Warmindo Well", "Kaze HQ", "Sadane Coffee",
        "Vion Coffee", "Yugata Coffee", "Warkop Pancong Lumer", "Talk Coffee Industri", "Kisah Lama Kopi",
        "Kaze HQ", "Yarra", "Its Brewme", "Ammo Coffee", "Hallo Burjois",
        "Umm A Yummy", "Cafe Loco", "Pasta Kangen Karawang", "Wardes", "Bebek Kaleyo Galuh Mas", "Riz Cafe dan Resto",
        "Sego Pedes Basman", "Soto Segerrr & Frozen", "Bubur & Soto Ayam Depan UNSIKA", "Mie Mukidi",
        "Warteg Putra", "Warteg Risky", "Warteg Kebumen"
    ],
    "Harga": [
        16547, 19350, 6756, 26283, 16966, 33120, 22037, 7711, 25549, 15684,
        26283, 25934, 15297, 20000, 22868,
        16510, 57255, 13798, 26247, 13245, 12666,
        6549, 8363, 5312, 9116, 6567, 5234, 5786
    ],
    "Waktu": [
        1, 4, 1, 1, 6, 2, 4, 1, 8, 8,
        1, 3, 4, 1, 4,
        1, 1, 3, 3, 10, 10,
        1, 1, 1, 2, 1, 1, 1
    ],
    "Suasana": [
        4, 3, 3, 5, 4, 4, 4, 1, 4, 2,
        5, 4, 5, 3, 5,
        3, 3, 2, 3, 5, 4,
        2, 3, 3, 2, 2, 2, 2
    ],
    "Kenyamanan": [
        4, 4, 3, 5, 4, 4, 4, 2, 4, 3,
        5, 5, 5, 3, 5,
        4, 3, 2, 3, 5, 4,
        2, 2, 3, 2, 3, 3, 3
    ],
    "Fasilitas": [
        4, 3, 3, 5, 3, 4, 3, 3, 4, 3,
        5, 5, 4, 3, 4,
        4, 2, 2, 4, 5, 5,
        2, 2, 2, 2, 3, 3, 3
    ]
}

df = pd.DataFrame(data)

# ------------------ FUZZIFIKASI ------------------ #
def buat_variabel_fuzzy():
    harga = ctrl.Antecedent(np.arange(5000, 35001, 1000), 'harga')
    waktu = ctrl.Antecedent(np.arange(1, 11, 1), 'waktu')
    suasana = ctrl.Antecedent(np.arange(1, 6, 1), 'suasana')
    kenyamanan = ctrl.Antecedent(np.arange(1, 6, 1), 'kenyamanan')
    fasilitas = ctrl.Antecedent(np.arange(1, 6, 1), 'fasilitas')
    rekomendasi = ctrl.Consequent(np.arange(0, 101, 1), 'rekomendasi')

    harga['murah'] = fuzz.trimf(harga.universe, [5000, 5000, 20000])
    harga['sedang'] = fuzz.trimf(harga.universe, [15000, 20000, 25000])
    harga['mahal'] = fuzz.trimf(harga.universe, [20000, 35000, 35000])

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

    return harga, waktu, suasana, kenyamanan, fasilitas, rekomendasi

# ------------------ RULE BASE ------------------ #
def buat_rules(harga, waktu, suasana, kenyamanan, fasilitas, rekomendasi):
    return [ 
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

def fuzzy_rekomendasi(harga_val, waktu_val, suasana_val, kenyamanan_val, fasilitas_val):
    harga, waktu, suasana, kenyamanan, fasilitas, rekomendasi = buat_variabel_fuzzy()
    rules = buat_rules(harga, waktu, suasana, kenyamanan, fasilitas, rekomendasi)

    sistem = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(sistem)

    sim.input['harga'] = harga_val
    sim.input['waktu'] = waktu_val
    sim.input['suasana'] = suasana_val
    sim.input['kenyamanan'] = kenyamanan_val
    sim.input['fasilitas'] = fasilitas_val

    sim.compute()
    return round(sim.output['rekomendasi'], 2), (harga, waktu, suasana, kenyamanan, fasilitas, rekomendasi)

# ------------------ PLOT FUZZY ------------------ #
def plot_keanggotaan(var, nama_var, nilai):
    fig, ax = plt.subplots()
    for term in var.terms:
        ax.plot(var.universe, var[term].mf, label=term)
        ax.vlines(nilai, 0, 1, colors='r', linestyles='dashed')
    ax.set_title(f"Derajat Keanggotaan: {nama_var}")
    ax.set_xlabel(nama_var)
    ax.set_ylabel("Keanggotaan")
    ax.legend()
    st.pyplot(fig)

# ------------------ STREAMLIT UI ------------------ #
st.set_page_config(page_title="Sistem Rekomendasi Tempat Kuliner", layout="wide")
st.title("🍽️ Sistem Rekomendasi Tempat Kuliner Berbasis Logika Fuzzy Mamdani")

# Pilih kategori
kategori = st.selectbox("Pilih Kategori Tempat kuliner:", df['Kategori'].unique())
df_kategori = df[df['Kategori'] == kategori].copy()

st.subheader("📋 Daftar Tempat Berdasarkan Kategori")
st.dataframe(df_kategori[['Nama Tempat', 'Kategori', 'Harga', 'Waktu', 'Suasana', 'Kenyamanan', 'Fasilitas']])

# Input preferensi user
with st.expander("🎛️ Masukkan Preferensi Anda"):
    harga_user = st.slider("Harga yang Diinginkan (Rp)", 5000, 35000, 20000, step=1000)
    waktu_user = st.slider("Waktu Tempuh", 1, 10, 3)
    suasana_user = st.slider("Suasana", 1, 5, 3)
    kenyamanan_user = st.slider("Kenyamanan", 1, 5, 3)
    fasilitas_user = st.slider("Fasilitas", 1, 5, 3)

    if st.button("🔍 Cari Rekomendasi"):
        skor_user, (harga, waktu, suasana, kenyamanan, fasilitas, rekomendasi) = fuzzy_rekomendasi(
            harga_user, waktu_user, suasana_user, kenyamanan_user, fasilitas_user
        )
        st.success(f"✅ Skor Rekomendasi Anda: **{skor_user}**")

        st.subheader("📈 Visualisasi Derajat Keanggotaan dari Input Anda")
        col1, col2, col3 = st.columns(3)
        with col1:
            plot_keanggotaan(harga, 'Harga', harga_user)
            plot_keanggotaan(waktu, 'Waktu', waktu_user)
        with col2:
            plot_keanggotaan(suasana, 'Suasana', suasana_user)
            plot_keanggotaan(kenyamanan, 'Kenyamanan', kenyamanan_user)
        with col3:
            plot_keanggotaan(fasilitas, 'Fasilitas', fasilitas_user)
            plot_keanggotaan(rekomendasi, 'Rekomendasi', skor_user)

        # Hitung skor semua tempat
        skor_tempat = []
        for _, row in df_kategori.iterrows():
            skor, _ = fuzzy_rekomendasi(
                row['Harga'], row['Waktu'], row['Suasana'],
                row['Kenyamanan'], row['Fasilitas']
            )
            skor_tempat.append(skor)

        df_kategori['Skor Rekomendasi'] = skor_tempat
        df_sorted = df_kategori.sort_values(by='Skor Rekomendasi', ascending=False)

        st.subheader("🏆 Rekomendasi Tempat Terbaik")
        st.dataframe(df_sorted[['Nama Tempat', 'Kategori', 'Skor Rekomendasi']].reset_index(drop=True))

        st.subheader("📊 Visualisasi Skor Semua Tempat")
        st.bar_chart(df_sorted.set_index('Nama Tempat')['Skor Rekomendasi'])
