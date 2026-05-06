import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pickle
import torch
from google import genai

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="SIKAP Muna Barat", page_icon="📂", layout="centered")

# --- LOAD MODEL & DATA VEKTOR ---
@st.cache_resource
def load_engine():
    return SentenceTransformer('intfloat/multilingual-e5-small')

@st.cache_data
def load_database():
    with open('database_sikap_vektor.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['dataframe'], data['embeddings']

# Menyalakan mesin E5
e5_model = load_engine()
df, corpus_embeddings = load_database()

# Pastikan vektor di CPU untuk menghindari error PyTorch
if not isinstance(corpus_embeddings, torch.Tensor):
    corpus_embeddings = torch.tensor(corpus_embeddings)
corpus_embeddings = corpus_embeddings.cpu()

# --- INISIALISASI KLIEN GEMINI ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.warning("⚠️ Peringatan: API Key belum dipasang di bagian Secrets Streamlit Cloud.")
    client = None

# --- ANTARMUKA PENGGUNA ---
st.title("📂 SIKAP AI")
st.subheader("Sistem Informasi Klasifikasi Arsip Pintar - Muna Barat")
st.markdown("---")

query_user = st.text_area("Masukkan perihal surat:", placeholder="Contoh: permohonan surat sertifikasi tanah untuk pembangunan gedung perpustakaan...")

if st.button("Cari Kode (AI Reasoning)", type="primary"):
    if query_user.strip() != "":
        if client is not None:
            with st.spinner('Tahap 1: AI merumuskan inti pencarian (Intent Extraction)...'):
                try:
                    # 1. RESEPSIONIS: Ekstrak kata kunci inti
                    prompt_ekstraksi = f"""
                    Ekstrak inti urusan dari perihal surat berikut dalam 1 sampai 3 kata saja untuk pencarian database.
                    Abaikan kata pengantar, lokasi, atau tujuan akhir (seperti: surat, permohonan, undangan, untuk, pembangunan, gedung).
                    Perihal: "{query_user}"
                    Format balasan: HANYA tulis kata kuncinya.
                    Contoh: "permohonan surat cuti tahunan untuk pegawai" -> "cuti tahunan"
                    """
                    
                    response_ekstraksi = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt_ekstraksi,
                    )
                    kata_kunci_inti = response_ekstraksi.text.strip().lower()
                    
                    st.info(f"🔍 Mesin mencari dengan kata kunci inti yang diekstrak: **{kata_kunci_inti}**")
                    
                    # 2. PENCARIAN VEKTOR MENGGUNAKAN KATA KUNCI INTI
                    query_text = f"query: {kata_kunci_inti}"
                    query_embedding = e5_model.encode(query_text, convert_to_tensor=True).cpu()
                    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                    top_results = torch.topk(cos_scores, k=10)
                    
                    kandidat_list = ""
                    for urutan, idx in enumerate(top_results.indices, 1):
                        res_idx = idx.item()
                        baris = df.iloc[res_idx]
                        kandidat_list += f"{urutan}. Kode: {baris['kode']} | Uraian: {baris['uraian']} | Jalur: {baris['breadcrumb']}\n"
                        
                except Exception as e:
                    st.error(f"Gagal mengekstrak kata kunci. Detail: {e}")
                    st.stop()
                    
            with st.spinner('Tahap 2: Hakim AI sedang memvonis jawaban terbaik...'):
                try:
                    # 3. HAKIM FINAL: Menentukan jawaban dari 10 kandidat yang sudah difilter
                    prompt_hakim = f"""
                    Kamu adalah Arsiparis Senior Muna Barat.
                    Pegawai mencari kode untuk urusan asli: "{query_user}"
                    
                    Berikut 10 kandidat kode terbaik dari database kami:
                    {kandidat_list}
                    
                    Tugasmu:
                    Pilih 1 kode yang paling tepat mewakili urusan asli pegawai.
                    Jika urusannya tentang pengadaan/sertifikasi lahan/tanah, pastikan pilih jalur Pertanahan.
                    
                    Format balasan:
                    KODE FINAL: [Tulis kodenya saja]
                    URAIAN: [Tulis uraian resminya]
                    ALASAN: [Jelaskan 2 kalimat mengapa ini yang paling tepat]
                    """
                    
                    response_hakim = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt_hakim,
                    )
                    
                    st.success("✨ Analisis Selesai!")
                    st.markdown(f"""
                    <div style="padding:20px; background-color:#f0f8ff; border-left: 5px solid #0056b3; border-radius: 5px;">
                        {response_hakim.text}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("Lihat 10 kandidat yang disortir mesin (Bahan Evaluasi)"):
                        st.text(kandidat_list)
                        
                except Exception as e:
                    st.error(f"Koneksi Hakim AI gagal. Detail: {e}")
        else:
            st.error("API Key belum terpasang di Secrets.")
    else:
        st.error("Silakan masukkan perihal surat terlebih dahulu.")
