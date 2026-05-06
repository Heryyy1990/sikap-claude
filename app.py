import streamlit as st
import json
import numpy as np
import re
import os
import time
import hashlib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Konfigurasi Halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIKAP – Klasifikasi Arsip Pintar",
    page_icon="🗂️",
    layout="centered",
)

# ─── CSS Kustom ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a3c5e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #5a7a9a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .rekomendasi-card {
        background: linear-gradient(135deg, #f0f6ff 0%, #e8f4fd 100%);
        border-left: 5px solid #1a6fbf;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .rekomendasi-rank {
        font-size: 0.75rem;
        font-weight: 700;
        color: #1a6fbf;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .rekomendasi-kode {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1a3c5e;
        font-family: 'Courier New', monospace;
    }
    .rekomendasi-uraian {
        font-size: 0.9rem;
        color: #2c4a6e;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    .rekomendasi-path {
        font-size: 0.75rem;
        color: #7a9ab5;
        margin-top: 0.4rem;
        font-style: italic;
    }
    .rekomendasi-score {
        font-size: 0.75rem;
        color: #3a8fbf;
        margin-top: 0.3rem;
    }
    .inti-box {
        background: #fff9e6;
        border-left: 4px solid #f5a623;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 1.2rem;
    }
    .inti-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #b07d1a;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .inti-text {
        font-size: 1.05rem;
        color: #3a2c00;
        font-weight: 600;
    }
    .step-badge {
        display: inline-block;
        background: #e8f0fe;
        color: #1967d2;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 6px;
    }
    .primer-badge {
        display: inline-block;
        background: #1a6fbf;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-weight: 700;
        font-family: monospace;
        margin-right: 6px;
    }
    .sekunder-badge {
        display: inline-block;
        background: #0d9488;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-weight: 700;
        font-family: monospace;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 6px;
        padding: 0.8rem 1rem;
    }
    .error-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 6px;
        padding: 0.8rem 1rem;
    }
    div[data-testid="stTextArea"] textarea {
        font-size: 1rem;
    }
    .footer-note {
        font-size: 0.72rem;
        color: #9ab0c5;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_data():
    with open(os.path.join(DATA_DIR, "klasifikasi_sikap_tree.json"), encoding="utf-8") as f:
        tree = json.load(f)
    with open(os.path.join(DATA_DIR, "klasifikasi_flat_lookup.json"), encoding="utf-8") as f:
        lookup = json.load(f)
    return tree, lookup

tree, lookup = load_data()

# ─── Helper Functions ─────────────────────────────────────────────────────────
def get_children_of(parent_kode: str, level_target: str) -> list[tuple[str, str, str]]:
    """Return list of (kode, uraian, search_text) for direct children at given level."""
    results = []
    for kode, data in lookup.items():
        if data["parent"] == parent_kode and data["level"] == level_target:
            results.append((kode, data["uraian"], data.get("search", data["uraian"])))
    return results

def tfidf_match(query: str, candidates: list[tuple], top_n: int = 3) -> list[tuple]:
    """
    TF-IDF cosine similarity match.
    candidates: list of (kode, uraian, search_text)
    Returns: list of (kode, uraian, score)
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        return [(candidates[0][0], candidates[0][1], 1.0)]

    texts = [c[2] for c in candidates]
    all_texts = [query] + texts
    try:
        vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1
        ).fit_transform(all_texts)
        sims = cosine_similarity(vec[0:1], vec[1:]).flatten()
    except Exception:
        # Fallback: simple word overlap
        query_words = set(query.lower().split())
        sims = np.array([
            len(query_words & set(t.lower().split())) / max(len(query_words), 1)
            for t in texts
        ])

    top_n = min(top_n, len(candidates))
    top_idx = sims.argsort()[::-1][:top_n]
    return [(candidates[i][0], candidates[i][1], float(sims[i])) for i in top_idx]

def build_primer_sekunder_text() -> str:
    """Build compact text listing all primer and sekunder for Gemini prompt."""
    lines = []
    for pkode, pdata in tree.items():
        lines.append(f"[PRIMER] {pkode}: {pdata['uraian']}")
        for skode, sdata in pdata["children"].items():
            lines.append(f"  [SEKUNDER] {skode}: {sdata['uraian']}")
    return "\n".join(lines)


# ─── Konstanta Rate Limiting ──────────────────────────────────────────────────
# Gemini Free Tier limits:
#   - 15 RPM  (requests per minute)  → min jeda 4 detik antar request
#   - 1.500 RPD (requests per day)
#   - Gemini 2.5 Flash Preview: lebih ketat, hanya 10 RPM
COOLDOWN_SECONDS = 6          # jeda minimum antar submit (detik)
RETRY_WAIT_SECONDS = 65       # tunggu setelah 429 sebelum retry
MAX_RETRIES_ON_429 = 2        # maks percobaan ulang setelah 429

# Cache hasil Gemini berdasarkan hash perihal → tidak habiskan kuota untuk input sama
@st.cache_data(ttl=3600, show_spinner=False)
def cached_gemini_call(perihal_hash: str, perihal: str, api_key_hash: str, _api_key: str) -> dict:
    """Wrapper cache untuk call_gemini. Cache 1 jam berdasarkan hash perihal."""
    return _call_gemini_raw(perihal, _api_key)

def _call_gemini_raw(perihal: str, api_key: str) -> dict:
    """
    Eksekusi REST call ke Gemini API dengan:
    - Auto-fallback 3 model (2.5-flash → 2.0-flash → 1.5-flash)
    - Auto-retry dengan countdown jika hit 429 (rate limit)
    """
    primer_sekunder_list = build_primer_sekunder_text()

    system_prompt = (
        "Anda adalah arsiparis ahli Pemerintah Kabupaten Muna Barat. "
        "Tugas Anda adalah menentukan kode klasifikasi arsip untuk surat yang masuk.\n\n"
        "TUGAS:\n"
        "1. Baca perihal/uraian surat.\n"
        "2. Tentukan INTI SURAT: frasa singkat (maks 8 kata) yang menangkap esensi pokok surat. "
        "Inti surat harus padat, tidak bertele-tele, dan langsung ke poin utama.\n"
        "3. Dari daftar KODE PRIMER (000–900), pilih 1 kode primer yang paling sesuai.\n"
        "4. Dari daftar KODE SEKUNDER di bawah primer terpilih, pilih 1 kode sekunder paling spesifik.\n\n"
        "ATURAN PENTING:\n"
        "- Kode sekunder WAJIB merupakan anak langsung dari primer terpilih (prefix sama).\n"
        "- Jawab HANYA dalam format JSON, tanpa teks tambahan:\n"
        '{"inti_surat":"...","primer":"000","sekunder":"000.1","alasan_primer":"...","alasan_sekunder":"..."}'
    )

    user_message = (
        f"PERIHAL SURAT:\n{perihal}\n\n"
        f"DAFTAR KODE KLASIFIKASI (PRIMER DAN SEKUNDER):\n{primer_sekunder_list}"
    )

    rest_payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {
            "maxOutputTokens": 400,
            "temperature": 0.1,
            "responseMimeType": "application/json"
        }
    }

    # Urutan model: 2.5-flash preview → 2.0-flash (stabil) → 1.5-flash (paling stabil)
    MODELS = [
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]

    for model_name in MODELS:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={api_key}"
        )
        retries_left = MAX_RETRIES_ON_429

        while retries_left >= 0:
            resp = requests.post(url, json=rest_payload, timeout=30)

            # 429 → Rate limit: tunggu dan retry
            if resp.status_code == 429:
                if retries_left > 0:
                    retries_left -= 1
                    # Tampilkan countdown di UI (countdown via Streamlit status)
                    wait_placeholder = st.empty()
                    for remaining in range(RETRY_WAIT_SECONDS, 0, -1):
                        wait_placeholder.warning(
                            f"⏳ **Kuota Gemini sementara penuh (rate limit).** "
                            f"Mencoba otomatis dalam **{remaining} detik**... "
                            f"(percobaan tersisa: {retries_left})"
                        )
                        time.sleep(1)
                    wait_placeholder.empty()
                    continue  # retry
                else:
                    # Habis retry, raise ke caller
                    raise requests.exceptions.HTTPError(response=resp)

            # 503/404 → model tidak tersedia, coba model berikutnya
            if resp.status_code in (503, 404):
                break  # keluar dari while, lanjut ke model berikutnya

            # Error lain (403, 500, dst)
            if not resp.ok:
                resp.raise_for_status()

            # Sukses → parse response
            data = resp.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            st.session_state["model_used"] = model_name
            raw_text = re.sub(r"^```json\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)
            return json.loads(raw_text)

    raise RuntimeError(
        "Semua model Gemini tidak dapat dihubungi (503/404). "
        "Kemungkinan Google AI Studio sedang gangguan. Coba beberapa menit lagi."
    )


def call_gemini(perihal: str, api_key: str) -> dict:
    """Entry point dengan cache. Perihal identik tidak akan re-call API."""
    # Hash perihal untuk cache key (tidak expose isi ke cache key secara langsung)
    perihal_hash = hashlib.md5(perihal.strip().lower().encode()).hexdigest()
    api_key_hash = hashlib.md5(api_key.encode()).hexdigest()
    return cached_gemini_call(perihal_hash, perihal.strip(), api_key_hash, api_key)



def local_classify(inti_surat: str, sekunder_kode: str, top_n: int = 3) -> list[dict]:
    """
    Local TF-IDF classification from sekunder down to kuartier.
    Returns top_n recommendations, each with full path.
    """
    recommendations = []

    # Get tersier children of chosen sekunder
    tersier_candidates = get_children_of(sekunder_kode, "tersier")
    if not tersier_candidates:
        # Fallback: sekunder itself is the result
        data = lookup.get(sekunder_kode, {})
        return [{
            "kode": sekunder_kode,
            "uraian": data.get("uraian", ""),
            "path": data.get("path", sekunder_kode),
            "level": "sekunder",
            "confidence": 0.5,
            "tersier_kode": None,
            "tersier_uraian": None,
        }]

    # Match inti surat to tersier
    top_tersier = tfidf_match(inti_surat, tersier_candidates, top_n=top_n)

    for t_kode, t_uraian, t_score in top_tersier:
        # Get kuartier children of this tersier
        kuartier_candidates = get_children_of(t_kode, "kuartier")

        if kuartier_candidates:
            top_k = tfidf_match(inti_surat, kuartier_candidates, top_n=1)
            if top_k:
                k_kode, k_uraian, k_score = top_k[0]
                combined_score = (t_score * 0.4 + k_score * 0.6)
                kdata = lookup.get(k_kode, {})
                recommendations.append({
                    "kode": k_kode,
                    "uraian": k_uraian,
                    "path": kdata.get("path", k_kode),
                    "level": "kuartier",
                    "confidence": combined_score,
                    "tersier_kode": t_kode,
                    "tersier_uraian": t_uraian,
                })
        else:
            # Fallback: tersier is the deepest
            tdata = lookup.get(t_kode, {})
            recommendations.append({
                "kode": t_kode,
                "uraian": t_uraian,
                "path": tdata.get("path", t_kode),
                "level": "tersier (fallback)",
                "confidence": t_score,
                "tersier_kode": t_kode,
                "tersier_uraian": t_uraian,
            })

    return recommendations[:top_n]

def confidence_bar(score: float) -> str:
    """Return color-coded confidence label."""
    if score >= 0.5:
        return f"🟢 Kesesuaian tinggi ({score:.0%})"
    elif score >= 0.25:
        return f"🟡 Kesesuaian sedang ({score:.0%})"
    else:
        return f"🔴 Kesesuaian rendah – perlu verifikasi ({score:.0%})"

# ─── Ambil API Key ────────────────────────────────────────────────────────────
gemini_api_key = None
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    pass

# ─── UI Utama ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🗂️ SIKAP</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Sistem Klasifikasi Arsip Pintar<br>'
    '<small>Pemerintah Kabupaten Muna Barat</small></div>',
    unsafe_allow_html=True,
)

# ─── Input API Key (jika belum ada di secrets) ────────────────────────────────
if not gemini_api_key:
    with st.expander("⚙️ Konfigurasi API Key Gemini", expanded=True):
        st.info(
            "Masukkan Google Gemini API Key Anda. "
            "Kunci ini tidak disimpan permanen. "
            "Untuk penggunaan rutin, tambahkan di **Settings → Secrets** sebagai `GEMINI_API_KEY`."
        )
        input_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
        if input_key:
            gemini_api_key = input_key

st.divider()

# ─── Form Input Perihal ───────────────────────────────────────────────────────
perihal_input = st.text_area(
    "📝 Masukkan Perihal / Uraian Surat",
    height=120,
    placeholder=(
        "Contoh: Permohonan Izin Pelaksanaan Kegiatan Sosialisasi Program Bantuan Operasional "
        "Sekolah (BOS) untuk Sekolah Dasar Negeri di Kecamatan Sawerigadi"
    ),
    help="Salin perihal surat atau tuliskan uraian singkat isi surat. Semakin detail, semakin akurat hasilnya.",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    cari = st.button("🔍 Tentukan Kode Klasifikasi", use_container_width=True, type="primary")

# ─── Proses Klasifikasi ───────────────────────────────────────────────────────
# Inisialisasi cooldown di session state
if "last_submit_time" not in st.session_state:
    st.session_state["last_submit_time"] = 0

if cari:
    if not perihal_input.strip():
        st.markdown('<div class="warning-box">⚠️ Mohon isi perihal surat terlebih dahulu.</div>', unsafe_allow_html=True)
    elif not gemini_api_key:
        st.markdown('<div class="error-box">🔑 API Key Gemini belum dikonfigurasi.</div>', unsafe_allow_html=True)
    else:
        # ── Cek cooldown antar submit ──────────────────────────────────────────
        elapsed = time.time() - st.session_state["last_submit_time"]
        if elapsed < COOLDOWN_SECONDS:
            sisa = int(COOLDOWN_SECONDS - elapsed) + 1
            st.warning(
                f"⏱️ Mohon tunggu **{sisa} detik** sebelum mengirim permintaan baru "
                f"(batas Gemini free tier: maks 10 request/menit)."
            )
            st.stop()

        st.session_state["last_submit_time"] = time.time()

        # ── Cek apakah hasil ini sudah di-cache ───────────────────────────────
        perihal_hash = hashlib.md5(perihal_input.strip().lower().encode()).hexdigest()
        is_cached = perihal_hash in st.session_state.get("cache_hits", set())

        with st.spinner("🤖 Menganalisis perihal surat dengan Gemini AI..."):
            try:
                gemini_result = call_gemini(perihal_input.strip(), gemini_api_key)
                # Tandai hash ini sebagai cached untuk session ini
                if "cache_hits" not in st.session_state:
                    st.session_state["cache_hits"] = set()
                st.session_state["cache_hits"].add(perihal_hash)

            except requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                if code == 429:
                    st.error(
                        "⏳ **Kuota harian/menit Gemini habis setelah percobaan ulang otomatis.**\n\n"
                        "Ini terjadi karena:\n"
                        "- **Free tier limit:** 10 RPM untuk Gemini 2.5 Flash Preview\n"
                        "- **1.500 request/hari** untuk semua model\n\n"
                        "**Solusi:**\n"
                        "- Tunggu 1–2 menit lalu coba lagi\n"
                        "- Jika sudah sering dipakai hari ini, tunggu hingga besok (reset pukul 00.00 UTC / 07.00 WITA)\n"
                        "- Pertimbangkan upgrade ke Gemini API berbayar jika kebutuhan tinggi"
                    )
                elif code == 403:
                    st.error(
                        "🔑 **API Key tidak valid atau tidak diizinkan.**\n\n"
                        "Pastikan:\n"
                        "- Key sudah benar (tidak ada spasi)\n"
                        "- Gemini API sudah diaktifkan di [Google AI Studio](https://aistudio.google.com)\n"
                        "- Key belum direvoke"
                    )
                elif code in (503, 404):
                    st.error(
                        f"❌ **Semua model Gemini tidak dapat dihubungi (HTTP {code}).**\n\n"
                        "Cek status layanan di [status.cloud.google.com](https://status.cloud.google.com). "
                        "Silakan coba beberapa menit lagi."
                    )
                else:
                    st.error(f"❌ Error HTTP {code}: {e}")
                st.stop()
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan tak terduga: {e}")
                st.stop()

        # Tampilkan info model dan status cache
        model_used = st.session_state.get("model_used", "—")
        cache_label = "♻️ Dari cache (tidak menggunakan kuota)" if is_cached else "✅ Baru dianalisis"
        st.caption(f"{cache_label} · Model: `{model_used}`")

        inti_surat = gemini_result.get("inti_surat", "").strip()
        primer_kode = gemini_result.get("primer", "").strip()
        sekunder_kode = gemini_result.get("sekunder", "").strip()
        alasan_primer = gemini_result.get("alasan_primer", "")
        alasan_sekunder = gemini_result.get("alasan_sekunder", "")

        # Validasi: pastikan sekunder adalah anak dari primer
        if not sekunder_kode.startswith(primer_kode + "."):
            # Cari sekunder terbaik secara lokal
            primer_data = tree.get(primer_kode, {})
            sekunder_list = list(primer_data.get("children", {}).keys())
            if sekunder_list:
                candidates = [(k, tree[primer_kode]["children"][k]["uraian"],
                               tree[primer_kode]["children"][k]["uraian"]) for k in sekunder_list]
                best = tfidf_match(inti_surat, candidates, top_n=1)
                if best:
                    sekunder_kode = best[0][0]

        # ── Tampilan Inti Surat & Pilihan Hirarki ──
        st.markdown(f"""
        <div class="inti-box">
            <div class="inti-label">💡 Inti Surat (AI)</div>
            <div class="inti-text">{inti_surat}</div>
        </div>
        """, unsafe_allow_html=True)

        primer_uraian = lookup.get(primer_kode, {}).get("uraian", "")
        sekunder_uraian = lookup.get(sekunder_kode, {}).get("uraian", "")

        col_p, col_s = st.columns(2)
        with col_p:
            st.markdown(
                f"**Kode Primer**  \n"
                f"<span class='primer-badge'>{primer_kode}</span> {primer_uraian}",
                unsafe_allow_html=True,
            )
        with col_s:
            st.markdown(
                f"**Kode Sekunder**  \n"
                f"<span class='sekunder-badge'>{sekunder_kode}</span> {sekunder_uraian}",
                unsafe_allow_html=True,
            )

        if alasan_sekunder:
            with st.expander("💬 Penjelasan AI untuk pilihan ini"):
                st.write(f"**Pemilihan Primer:** {alasan_primer}")
                st.write(f"**Pemilihan Sekunder:** {alasan_sekunder}")

        st.divider()

        # ── Klasifikasi Lokal (Tersier + Kuartier) ──
        with st.spinner("📂 Mencocokkan kode tersier dan kuartier secara lokal..."):
            recs = local_classify(inti_surat, sekunder_kode, top_n=3)

        if not recs:
            st.warning("Tidak ditemukan kode yang cocok. Coba perjelas perihal surat.")
        else:
            st.markdown("### 📋 Rekomendasi Kode Klasifikasi")
            rank_labels = ["🥇 Rekomendasi Utama", "🥈 Rekomendasi Alternatif 1", "🥉 Rekomendasi Alternatif 2"]

            for i, rec in enumerate(recs):
                label = rank_labels[i] if i < len(rank_labels) else f"Alternatif {i+1}"
                level_label = rec["level"].replace("tersier (fallback)", "Tersier ⚠️ (tidak ada kuartier)")

                # Format path untuk tampilan
                path_parts = rec["path"].split(" > ")
                path_display = " → ".join(path_parts)

                st.markdown(f"""
                <div class="rekomendasi-card">
                    <div class="rekomendasi-rank">{label}</div>
                    <div class="rekomendasi-kode">{rec['kode']}</div>
                    <div class="rekomendasi-uraian">{rec['uraian'].title()}</div>
                    <div class="rekomendasi-path">📍 {path_display}</div>
                    <div class="rekomendasi-score">{confidence_bar(rec['confidence'])} &nbsp;|&nbsp; Level: {level_label}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Catatan Arsiparis ──
            if any(rec["confidence"] < 0.2 for rec in recs):
                st.markdown(
                    '<div class="warning-box">'
                    "⚠️ <b>Catatan:</b> Tingkat kesesuaian beberapa rekomendasi rendah. "
                    "Harap verifikasi secara manual dengan tabel klasifikasi lengkap."
                    "</div>",
                    unsafe_allow_html=True,
                )

st.markdown("---")

# ─── Panel Eksplorasi (Sidebar) ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔎 Jelajahi Kode Klasifikasi")
    st.caption("Cari kode klasifikasi secara manual untuk verifikasi.")

    search_query = st.text_input("Cari uraian...", placeholder="ketik kata kunci")

    if search_query and len(search_query) >= 3:
        all_candidates = [
            (k, v["uraian"], v.get("search", v["uraian"]))
            for k, v in lookup.items()
        ]
        results = tfidf_match(search_query, all_candidates, top_n=10)
        if results:
            for kode, uraian, score in results:
                lv = lookup[kode]["level"]
                indent = "　" * (len(kode.split(".")) - 1)
                st.markdown(
                    f"`{kode}` {indent}{uraian}  \n"
                    f"<small style='color:gray'>Level: {lv} | Skor: {score:.2f}</small>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Tidak ditemukan.")

    st.divider()
    st.markdown("### 📊 Info Dataset")
    primer_count = sum(1 for v in lookup.values() if v["level"] == "primer")
    sekunder_count = sum(1 for v in lookup.values() if v["level"] == "sekunder")
    tersier_count = sum(1 for v in lookup.values() if v["level"] == "tersier")
    kuartier_count = sum(1 for v in lookup.values() if v["level"] == "kuartier")
    st.metric("Total Kode", f"{len(lookup):,}")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Primer", primer_count)
        st.metric("Tersier", f"{tersier_count:,}")
    with col_b:
        st.metric("Sekunder", sekunder_count)
        st.metric("Kuartier", f"{kuartier_count:,}")

    st.divider()
    st.caption(
        "**SIKAP** – Sistem Klasifikasi Arsip Pintar  \n"
        "Pemkab Muna Barat · Powered by Gemini AI + TF-IDF"
    )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer-note">'
    "Aplikasi ini adalah alat bantu arsiparis. Keputusan akhir kode klasifikasi "
    "tetap menjadi tanggung jawab arsiparis berdasarkan peraturan yang berlaku."
    "</div>",
    unsafe_allow_html=True,
)
