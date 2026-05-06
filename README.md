# 🗂️ SIKAP – Sistem Klasifikasi Arsip Pintar
**Pemerintah Kabupaten Muna Barat**

Aplikasi bantu arsiparis untuk menentukan kode klasifikasi arsip secara cerdas menggunakan kombinasi Gemini AI (untuk memahami inti surat) dan TF-IDF lokal (untuk penelusuran kode tersier & kuartier).

---

## 📁 Struktur Proyek

```
sikap_app/
├── app.py                          # Aplikasi Streamlit utama
├── requirements.txt                # Dependensi Python
├── .gitignore
├── .streamlit/
│   └── secrets.toml               # API Key (JANGAN push ke GitHub!)
└── data/
    ├── klasifikasi_sikap_tree.json     # Struktur pohon hierarki kode
    ├── klasifikasi_flat_lookup.json    # Tabel lookup flat untuk pencarian cepat
    └── klasifikasi_sikap_enriched.csv  # Dataset lengkap dengan konteks pengayaan
```

---

## 🚀 Cara Menjalankan (Lokal)

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Buat file secrets
mkdir -p .streamlit
echo 'GEMINI_API_KEY = "AIzaSy..."' > .streamlit/secrets.toml

# 3. Jalankan
streamlit run app.py
```

---

## ☁️ Deploy ke Streamlit Community Cloud

1. Push semua file ke GitHub (pastikan `.streamlit/secrets.toml` di `.gitignore`)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect repository → pilih `app.py`
4. **Settings → Secrets**, isi:
   ```
   GEMINI_API_KEY = "AIzaSy..."
   ```
5. Deploy!

---

## 🔧 Arsitektur Teknis

| Komponen | Teknologi | Keterangan |
|---|---|---|
| Antarmuka | Streamlit (gratis) | UI web interaktif |
| AI Utama | Gemini 2.5 Flash (gratis) | Ekstrak inti surat + pilih primer & sekunder |
| Matching Lokal | TF-IDF (scikit-learn) | Cocokkan tersier & kuartier tanpa API |
| Dataset | JSON + CSV | 2.829 kode klasifikasi, 4 level hierarki |

### Alur Kerja
```
[Perihal Surat Input]
       ↓
[Gemini AI] ← 1 API call saja
   → Ekstrak Inti Surat (maks 8 kata)
   → Pilih Kode Primer (dari 10 opsi)  
   → Pilih Kode Sekunder (dari ~1-17 opsi)
       ↓
[TF-IDF Lokal] ← tanpa API, bebas token
   → Cocokkan Kode Tersier (dari anak sekunder)
   → Cocokkan Kode Kuartier (dari anak tersier)
       ↓
[3 Rekomendasi Kode Klasifikasi]
```

---

## ⚠️ Keterbatasan

- **Gemini Free Tier:** 15 RPM, 1.500 request/hari → cukup untuk penggunaan normal
- **Kode duplikat:** 272 uraian yang sama di level berbeda → diatasi dengan SearchText (konteks induk)
- Keputusan final tetap wewenang arsiparis

---

## 📊 Statistik Dataset

| Level | Jumlah Kode |
|---|---|
| Primer (000–900) | 10 |
| Sekunder | 54 |
| Tersier | 653 |
| Kuartier | 2.112 |
| **Total** | **2.829** |
