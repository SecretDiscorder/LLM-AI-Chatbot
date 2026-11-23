import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

OLLAMA_URL = "http://localhost:11434"
MODELS_DEFAULT = ["llama3.2:1b", "qwen2.5:7b"]
BLOOM = {
    "C1 — Mengingat": "Mengingat fakta, istilah, nama, atau definisi.",
    "C2 — Memahami": "Menjelaskan makna dengan kata sendiri, merangkum, atau mengklasifikasikan.",
    "C3 — Menerapkan": "Menggunakan konsep atau prosedur pada kasus tertentu.",
    "C4 — Menganalisis": "Mengurai hubungan, sebab-akibat, perbandingan, dan struktur.",
    "C5 — Mengevaluasi": "Menilai klaim atau keputusan berdasarkan kriteria dan bukti.",
    "C6 — Mencipta": "Menghasilkan rancangan, solusi, model, atau gagasan baru.",
}
SYSTEM_PROMPT = """Kamu adalah asisten pengetahuan yang mengutamakan ketepatan maksud pertanyaan.

ATURAN WAJIB:
1. Identifikasi inti pertanyaan dan jawab inti tersebut pada kalimat pertama.
2. Jangan mendahulukan latar belakang, tokoh terkait, atau sejarah umum sebelum jawaban inti.
3. Bedakan secara tegas: penemu, perintis, pengembang, penyempurna, dan pengomersial.
4. Jika suatu teknologi tidak memiliki satu penemu tunggal, katakan demikian lalu sebutkan tokoh yang kontribusinya paling relevan.
5. Jangan mengarang. Nyatakan ketidakpastian bila bukti tidak cukup.
6. Tambahkan konteks hanya setelah jawaban langsung.
7. Ikuti tingkat kognitif yang diminta, tetapi jangan mengubah pertanyaan pengguna.

FORMAT:
Jawaban inti: [jawaban langsung]
Penjelasan: [alasan singkat]
Konteks tambahan: [opsional, hanya jika membantu]
"""
EVAL_PROMPT = """Nilai jawaban model berdasarkan pertanyaan pengguna.
Keluarkan JSON valid saja dengan struktur:
{
 "intent_match": 1-5,
 "factual_quality": 1-5,
 "directness": 1-5,
 "conceptual_precision": 1-5,
 "bloom_alignment": 1-5,
 "notes": "alasan singkat",
 "possible_issue": "kesalahan atau hal yang perlu diverifikasi; string kosong jika tidak ada"
}
Jangan memberi nilai tinggi hanya karena jawaban panjang. Jawaban harus menjawab inti pertanyaan terlebih dahulu.
Ini adalah evaluasi awal, bukan verifikasi fakta independen.
"""

st.set_page_config(page_title="Local LLM Prioritizer", page_icon="🧠", layout="wide")
st.title("🧠 Local LLM Evaluation & Answer Prioritization")
st.caption("Ollama · Llama 3.2 1B · Qwen 2.5 7B · Taksonomi Bloom Revisi")

with st.sidebar:
    st.header("Pengaturan")
    ollama_url = st.text_input("Ollama URL", OLLAMA_URL).rstrip("/")
    try:
        tags = requests.get(f"{ollama_url}/api/tags", timeout=3).json().get("models", [])
        installed = [m.get("name") for m in tags]
    except Exception:
        installed = []
        st.warning("Ollama tidak terhubung. Jalankan `ollama serve`.")
    st.write("Model terdeteksi:")
    if installed:
        for m in installed:
            st.write("• " + m)
    else:
        st.caption("Belum ada model terdeteksi.")
    model_a = st.selectbox("Model A", MODELS_DEFAULT, index=0)
    model_b = st.selectbox("Model B", MODELS_DEFAULT, index=1)
    use_system = st.checkbox("Gunakan prompt prioritas jawaban", value=True)
    show_eval = st.checkbox("Jalankan evaluator otomatis", value=True)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

tab1, tab2, tab3 = st.tabs(["Uji pertanyaan", "Benchmark", "Panduan"])

def ollama_chat(model, messages, temp=0.2):
    response = requests.post(
        f"{ollama_url}/api/chat",
        json={"model": model, "messages": messages, "stream": False,
              "options": {"temperature": temp}},
        timeout=300
    )
    response.raise_for_status()
    return response.json()["message"]["content"]

def evaluate(question, answer, bloom):
    prompt = f"""Tingkat Bloom yang diminta: {bloom}
Pertanyaan: {question}
Jawaban model:
{answer}
"""
    raw = ollama_chat(model_b, [
        {"role": "system", "content": EVAL_PROMPT},
        {"role": "user", "content": prompt}
    ], 0.0)
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return {"notes": raw, "possible_issue": "Evaluator tidak menghasilkan JSON valid."}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"notes": raw, "possible_issue": "JSON evaluator gagal diparse."}

with tab1:
    question = st.text_area("Pertanyaan", "Siapa tokoh utama dalam pengembangan sistem listrik AC dan DC?", height=100)
    bloom = st.selectbox("Tingkat Taksonomi Bloom", list(BLOOM.keys()), index=0)
    st.caption(BLOOM[bloom])
    custom_context = st.text_area("Konteks tambahan (opsional)", placeholder="Misalnya: jawab untuk konteks sejarah teknologi.")
    run = st.button("Jalankan perbandingan", type="primary", use_container_width=True)
    if run and question.strip():
        system = SYSTEM_PROMPT if use_system else "Jawab pertanyaan pengguna dengan akurat."
        user_prompt = f"Tingkat kognitif: {bloom}\n{BLOOM[bloom]}\n\nPertanyaan: {question}"
        if custom_context.strip():
            user_prompt += "\n\nKonteks tambahan: " + custom_context.strip()
        outputs = []
        cols = st.columns(2)
        for col, model in zip(cols, [model_a, model_b]):
            with col:
                st.subheader(model)
                start = time.time()
                try:
                    answer = ollama_chat(model, [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt}
                    ], temperature)
                    elapsed = time.time() - start
                    st.markdown(answer)
                    st.caption(f"Waktu respons: {elapsed:.2f} detik")
                    outputs.append((model, answer, elapsed))
                except Exception as e:
                    st.error(f"Gagal menjalankan {model}: {e}")
                    outputs.append((model, "", 0))
        if show_eval and len(outputs) == 2 and all(x[1] for x in outputs):
            st.divider()
            st.subheader("Evaluasi otomatis")
            st.warning("Skor evaluator berbasis model; gunakan pemeriksaan manusia/sumber tepercaya untuk validasi fakta.")
            eval_cols = st.columns(2)
            for col, (model, answer, elapsed) in zip(eval_cols, outputs):
                with col:
                    st.markdown(f"**{model}**")
                    try:
                        score = evaluate(question, answer, bloom)
                        numeric = [score.get(k, 0) for k in ["intent_match","factual_quality","directness","conceptual_precision","bloom_alignment"]]
                        if all(isinstance(v, (int, float)) for v in numeric):
                            st.metric("Rata-rata skor / 5", f"{sum(numeric)/5:.2f}")
                        st.json(score)
                    except Exception as e:
                        st.error(f"Evaluasi gagal: {e}")
            record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "question": question, "bloom": bloom,
                "models": [{"model": m, "answer": a, "seconds": t} for m,a,t in outputs]
            }
            Path("data").mkdir(exist_ok=True)
            with open("data/history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            st.success("Hasil perbandingan tersimpan di data/history.jsonl")

with tab2:
    st.subheader("Benchmark pertanyaan dasar")
    samples = [
        ("C1 — Mengingat", "Siapa tokoh utama pengembangan sistem listrik AC dan DC?"),
        ("C2 — Memahami", "Jelaskan perbedaan arus AC dan DC dengan bahasa sederhana."),
        ("C3 — Menerapkan", "Adaptor mengubah listrik AC rumah menjadi DC. Jelaskan proses dasarnya."),
        ("C4 — Menganalisis", "Bandingkan kelebihan dan keterbatasan distribusi listrik AC dan DC."),
        ("C5 — Mengevaluasi", "Evaluasi klaim bahwa Tesla adalah satu-satunya penemu listrik AC."),
        ("C6 — Mencipta", "Rancang eksperimen sederhana yang aman untuk menunjukkan perbedaan AC dan DC.")
    ]
    st.write("Gunakan pertanyaan ini untuk melihat konsistensi model di enam tingkat Bloom.")
    for level, q in samples:
        with st.expander(level):
            st.write(q)
            if st.button("Pakai pertanyaan ini", key=level):
                st.session_state["benchmark_question"] = q
                st.info("Salin pertanyaan ke tab Uji pertanyaan.")

with tab3:
    st.subheader("Cara menjalankan")
    st.code("""# 1. Pastikan Ollama berjalan
ollama serve

# 2. Unduh model (jika belum ada)
ollama pull llama3.2:1b
ollama pull qwen2.5:7b

# 3. Buat virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Pasang dependensi
pip install -r requirements.txt

# 5. Jalankan aplikasi
streamlit run app.py
""", language="bash")
    st.markdown("**Catatan:** evaluator memakai Qwen sebagai penilai bila Model B dipilih sebagai Qwen. Untuk penilaian lebih kuat, gunakan evaluator terpisah yang lebih besar atau verifikasi manusia.")
    st.markdown("Aplikasi ini tidak melakukan pencarian web/RAG. Ia menguji perilaku model berdasarkan pengetahuan internalnya.")

st.divider()
st.caption("Local-first prototype · Data pertanyaan disimpan lokal dalam data/history.jsonl")
