import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pickle
import torch

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="SIKAP Muna Barat",
    page_icon="📂",
    layout="centered"
)

# --- STYLE CSS CUSTOM (Agar lebih profesional) ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .result-card {
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        background-color: white;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL & DATA (Dichace agar cepat) ---
@st.cache_resource
def load_engine():
    # Memuat model ringan E5-Small
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    return model

@st.cache_data
def load_database():
    # Memuat hasil kerja keras dari Colab (.pkl)
    with open('database_sikap_vektor.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['dataframe'], data['embeddings']

# Inisialisasi
model = load_engine()
df, corpus_embeddings = load_database()

# --- INTERFACE PENGGUNA ---
st.title("📂 SIKAP")
st.subheader("Sistem Informasi Klasifikasi Arsip Pintar")
st.info("Kabupaten Muna Barat")

st.markdown("---")
st.write("Masukkan perihal surat atau ringkasan isi surat di bawah ini:")

# Input User
query_user = st.text_area("", placeholder="Contoh: permohonan bantuan bibit jagung untuk kelompok tani...", height=100)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    btn_cari = st.button("Cari Kode Klasifikasi")

# --- LOGIKA PENCARIAN ---
if btn_cari:
    if query_user.strip() != "":
        with st.spinner('AI sedang menganalisis kode yang tepat...'):
            # E5 Requirement: Query harus diawali dengan 'query: '
            query_text = f"query: {query_user}"
            query_embedding = model.encode(query_text, convert_to_tensor=True)
            
            # Hitung kemiripan dengan database
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            
            # Ambil 3 hasil terbaik
            top_results = torch.topk(cos_scores, k=3)
            
            st.markdown("### Hasil Analisis AI:")
            
            for score, idx in zip(top_results.values, top_results.indices):
                res_idx = idx.item()
                row = df.iloc[res_idx]
                score_pct = score.item() * 100
                
                # Tampilkan dalam bentuk card
                st.markdown(f"""
                <div class="result-card">
                    <span style="color: #007bff; font-weight: bold; font-size: 1.2em;">{row['kode']}</span><br>
                    <span style="font-weight: 500;">{row['uraian'].upper()}</span><br>
                    <small style="color: #6c757d;">Jalur: {row['breadcrumb']}</small><br>
                    <div style="text-align: right;">
                        <span style="background-color: #e7f3ff; color: #007bff; padding: 2px 8px; border-radius: 10px; font-size: 0.8em;">
                            Akurasi: {score_pct:.1f}%
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    else:
        st.error("Silakan masukkan teks perihal terlebih dahulu.")

# --- FOOTER ---
st.markdown("---")
st.caption("© 2026 SIKAP - Dinas Perpustakaan dan Kearsipan Muna Barat")
