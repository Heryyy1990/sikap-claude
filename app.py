import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pickle
import torch
from google import genai
import re

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

e5_model = load_engine()
df, corpus_embeddings = load_database()

if not isinstance(corpus_embeddings, torch.Tensor):
    corpus_embeddings = torch.tensor(corpus_embeddings)
corpus_embeddings = corpus_embeddings.cpu()

# --- INISIALISASI KLIEN GEMINI ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.warning("⚠️ Peringatan: API Key belum dipasang di bagian Secrets.")
    client = None

# --- ANTARMUKA PENGGUNA ---
st.title("📂 SIKAP AI")
st.subheader("Sistem Informasi Klasifikasi Arsip Pintar - Muna Barat")
st.markdown("---")

query_user = st.text_area("Masukkan perihal surat:", placeholder="Contoh: permohonan surat sertifikasi tanah untuk gedung...")

if st.button("Cari Kode (AI Reasoning)", type="primary"):
    if query_user.strip() != "":
        if client is not None:
            with st.spinner('Tahap 1: Mengekstrak kata kunci inti...'):
                try:
                    # 1. RESEPSIONIS: Ekstrak kata kunci
                    prompt_ekstraksi = f"""
                    Ekstrak inti urusan dari perihal surat ini dalam 1-3 kata saja.
                    Abaikan kata: surat, permohonan, undangan, untuk, pembangunan, gedung.
                    Perihal: "{query_user}"
                    """
                    response_ekstraksi = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt_ekstraksi,
                    )
                    kata_kunci_inti = response_ekstraksi.text.strip().lower()
                    
                    # Bersihkan tanda baca
                    kata_kunci_inti = re.sub(r'[^\w\s]', '', kata_kunci_inti)
                    st.info(f"🔍 Mesin mencari dengan kata kunci: **{kata_kunci_inti}**")
                    
                    kandidat_final = []
                    kandidat_teks = ""
                    nomor = 1

                    # --- JARING 1: LEKSIKAL (Pencocokan Kata Persis) ---
                    # Cari baris yang mengandung kata-kata tersebut di uraian atau jalur
                    kata_list = kata_kunci_inti.split()
                    if kata_list:
                        # Buat regex untuk mencari salah satu kata inti
                        pola_pencarian = '|'.join(kata_list)
                        df_leksikal = df[df['breadcrumb'].str.contains(pola_pencarian, case=False, na=False) | 
                                         df['uraian'].str.contains(pola_pencarian, case=False, na=False)].head(5)
                        
                        for idx, baris in df_leksikal.iterrows():
                            if baris['kode'] not in [k['kode'] for k in kandidat_final]:
                                kandidat_final.append(baris)
                                kandidat_teks += f"{nomor}. [JARING KATA] Kode: {baris['kode']} | Uraian: {baris['uraian']} | Jalur: {baris['breadcrumb']}\n"
                                nomor += 1

                    # --- JARING 2: SEMANTIK (Vektor E5) ---
                    query_text = f"query: {kata_kunci_inti}"
                    query_embedding = e5_model.encode(query_text, convert_to_tensor=True).cpu()
                    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                    top_results = torch.topk(cos_scores, k=10)
                    
                    for idx in top_results.indices:
                        res_idx = idx.item()
                        baris = df.iloc[res_idx]
                        if baris['kode'] not in [k['kode'] for k in kandidat_final] and nomor <= 10:
                            kandidat_final.append(baris)
                            kandidat_teks += f"{nomor}. [JARING VEKTOR] Kode: {baris['kode']} | Uraian: {baris['uraian']} | Jalur: {baris['breadcrumb']}\n"
                            nomor += 1
                            
                except Exception as e:
                    st.error(f"Gagal mencari kandidat. Detail: {e}")
                    st.stop()
                    
            with st.spinner('Tahap 2: Hakim AI sedang memvonis...'):
                try:
                    # 3. HAKIM FINAL
                    prompt_hakim = f"""
                    Kamu adalah Arsiparis Senior Muna Barat.
                    Pegawai mencari kode untuk urusan: "{query_user}"
                    
                    Berikut 10 kandidat kode dari database kami:
                    {kandidat_teks}
                    
                    Tugasmu:
                    Pilih 1 kode yang paling akurat mewakili urusan asli pegawai.
                    Jika terkait lahan/tanah/sertifikat tanah, pilih jalur Pertanahan (biasanya 500.17).
                    
                    Format balasan:
                    KODE FINAL: [Tulis kodenya saja]
                    URAIAN: [Tulis uraian resminya]
                    ALASAN: [Jelaskan singkat mengapa ini yang paling tepat]
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
                        st.text(kandidat_teks)
                        
                except Exception as e:
                    st.error(f"Koneksi Hakim AI gagal. Detail: {e}")
        else:
            st.error("API Key belum terpasang di Secrets.")
    else:
        st.error("Silakan masukkan perihal surat terlebih dahulu.")
