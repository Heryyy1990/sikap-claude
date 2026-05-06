import pandas as pd
from sentence_transformers import util
import torch
import pickle

# 1. Buka otak yang sudah dibakar
with open('database_sikap_vektor.pkl', 'rb') as f:
    data = pickle.load(f)
    
df = data['dataframe']
corpus_embeddings = data['embeddings']

# 2. Kalimat uji yang paling susah dan bikin ngawur tadi
query_user = "permohonan surat sertifikasi tanah untuk pembangunan gedung perpustakaan"
query_text = f"query: {query_user}"

# (Asumsi model E5 sudah ada di memori Colab Anda dari script sebelumnya)
query_embedding = model.encode(query_text, convert_to_tensor=True)

# 3. Hitung jarak kemiripan
cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

# 4. AMBIL 20 KANDIDAT TERATAS (Ini yang akan dikirim ke Gemini nanti)
top_results = torch.topk(cos_scores, k=20)

print(f"Hasil Top-20 untuk: '{query_user}'\n")

ditemukan = False
for urutan, (score, idx) in enumerate(zip(top_results.values, top_results.indices), 1):
    res_idx = idx.item()
    kode = df.iloc[res_idx]['kode']
    uraian = df.iloc[res_idx]['uraian']
    
    # Beri tanda bintang jika ini adalah kode Pertanahan (500.17.x)
    tanda = "⭐⭐⭐" if "500.17" in str(kode) else ""
    if "500.17" in str(kode): ditemukan = True
        
    print(f"Rank {urutan} | Skor: {score.item()*100:.1f}% | Kode: {kode} | Uraian: {uraian} {tanda}")

print("\nKESIMPULAN:")
if ditemukan:
    print("✅ BERHASIL! Kode Pertanahan masuk ke dalam Top-20. Hakim Gemini PASTI bisa menemukannya!")
else:
    print("❌ GAGAL! Kode Pertanahan terlempar dari Top-20. Data yang dikirim ke Gemini akan menjadi sampah.")
