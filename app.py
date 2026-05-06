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

# --- INISIALISASI KLIEN GEMINI ---
# Mengambil API Key dari brankas Streamlit
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

query_user = st.text_area("Masukkan perihal surat:", placeholder="Contoh: permohonan sertifikasi tanah untuk gedung perpustakaan...")

if st.button("Cari Kode (AI Reasoning)", type="primary"):
    if query_user.strip() != "":
        with st.spinner('Langkah 1: Menyortir ribuan dokumen...'):
            # 1. E5 mengambil 10 kandidat terbaik
            query_text = f"query: {query_user}"
            query_embedding = e5_model.encode(query_text, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=10)
            
            # Susun daftar kandidat
            kandidat_list = ""
            for urutan, idx in enumerate(top_results.indices, 1):
                res_idx = idx.item()
                baris = df.iloc[res_idx]
                kandidat_list += f"{urutan}. Kode: {baris['kode']} | Uraian: {baris['uraian']} | Jalur: {baris['breadcrumb']}\n"
                
        with st.spinner('Langkah 2: Hakim AI (Gemini 2.5 Flash) sedang menalar...'):
            if client is not None:
                try:
                    # 2. Instruksi ketat untuk Gemini 2.5 Flash
                    prompt = f"""
                    Kamu adalah Arsiparis Senior di Pemerintahan Kabupaten Muna Barat.
                    Seorang pegawai menanyakan kode klasifikasi arsip untuk surat tentang: "{query_user}"
                    
                    Mesin pencari kami menemukan 10 kandidat kode berdasarkan kedekatan teks:
                    {kandidat_list}
                    
                    Tugasmu:
                    1. Analisis perihal surat pegawai tersebut.
                    2. Pilih SATU kode klasifikasi yang paling akurat dari 10 kandidat di atas.
                    3. Jika surat terkait sertifikasi/pengadaan tanah untuk bangunan pemerintah, pilih urusan Pertanahan, bukan sertifikasi profesi atau pinjaman uang.
                    
                    Format balasanmu persis seperti ini (tanpa basa-basi):
                    KODE FINAL: [Tulis kodenya saja, misal 500.17.3]
                    URAIAN: [Tulis uraian resminya]
                    ALASAN: [Jelaskan secara singkat mengapa kode ini paling tepat dan mengapa kandidat lain salah]
                    """
                    
                    # Memanggil Gemini menggunakan SDK google-genai terbaru
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                    )
                    
                    hasil_hakim = response.text
                    
                    # Tampilkan hasil akhir
                    st.success("✨ Analisis Selesai!")
                    st.markdown(f"""
                    <div style="padding:20px; background-color:#f0f8ff; border-left: 5px solid #0056b3; border-radius: 5px;">
                        {hasil_hakim}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("Lihat 10 kandidat yang disortir mesin E5 (Bahan Evaluasi)"):
                        st.text(kandidat_list)
                        
                except Exception as e:
                    st.error(f"Koneksi ke Gemini gagal. Detail error: {e}")
            else:
                st.error("Sistem Hakim AI tidak dapat berjalan karena API Key belum dikonfigurasi.")
    else:
        st.error("Silakan masukkan perihal surat terlebih dahulu.")
