import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

OLLAMA_URL = "http://localhost:11434"
MODELS_DEFAULT = ["llama3.2:1b", "qwen3.5:4b", "gemma3:1b"]

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

STRICT_GROUNDING_ADDON = """
ATURAN REFERENSI KETAT (WAJIB DIPATUHI):
8. Jawab HANYA berdasarkan teks pada bagian 'Referensi Wajib'. Abaikan pengetahuan luar jika bertentangan atau tidak ada di referensi.
9. Jika jawaban tidak tersedia secara eksplisit di dalam referensi, jawab persis: "Informasi tidak ditemukan dalam referensi yang diberikan."
10. Sertakan kutipan singkat dari referensi untuk mendukung jawaban inti.
"""

EVAL_PROMPT = """Nilai jawaban model berdasarkan pertanyaan pengguna dan referensi (jika tersedia).
Keluarkan JSON valid saja dengan struktur:
{
 "intent_match": 1-5,
 "factual_quality": 1-5,
 "directness": 1-5,
 "conceptual_precision": 1-5,
 "bloom_alignment": 1-5,
 "hallucination_risk": "Rendah | Sedang | Tinggi",
 "unsupported_claims": ["daftar klaim spesifik yang meragukan atau tidak ada di referensi"],
 "notes": "alasan singkat",
 "possible_issue": "kesalahan atau hal yang perlu diverifikasi; string kosong jika tidak ada"
}
Jangan memberi nilai tinggi hanya karena jawaban panjang. Jawaban harus menjawab inti pertanyaan terlebih dahulu.
Ini adalah evaluasi awal, bukan verifikasi fakta independen.
"""

st.set_page_config(page_title="Local LLM Prioritizer", page_icon="🧠", layout="wide")
st.title("🧠 Local LLM Evaluation & Answer Prioritization")
st.caption("Ollama · Anti-Halusinasi & Grounding · Taksonomi Bloom Revisi")

with st.sidebar:
    st.header("Pengaturan Model")
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

    available_models = list(dict.fromkeys(installed + MODELS_DEFAULT))
    idx_b = 1 if len(available_models) > 1 else 0

    model_a = st.selectbox("Model A", available_models, index=0)
    model_b = st.selectbox("Model B", available_models, index=idx_b)
    eval_model = st.selectbox(
        "Model Evaluator (Juri)",
        available_models,
        index=idx_b,
        help="Sebaiknya pilih model dengan parameter terbesar untuk mengurangi bias penilaian."
    )

    st.divider()
    st.header("Kontrol Anti-Halusinasi")
    use_system = st.checkbox("Gunakan prompt prioritas jawaban", value=True)
    strict_grounding = st.checkbox(
        "Mode Referensi Ketat (Strict Grounding)",
        value=False,
        help="Memaksa model hanya menjawab dari teks/file referensi yang diberikan."
    )
    show_eval = st.checkbox("Jalankan evaluator otomatis", value=True)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    top_p = st.slider("Top-P (Fokus Token)", 0.1, 1.0, 0.85, 0.05)

tab1, tab2, tab3 = st.tabs(["Uji pertanyaan", "Benchmark", "Panduan"])


def ollama_chat(model, messages, temp=0.1, top_p_val=0.85, json_mode=False):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temp,
            "top_p": top_p_val,
            "repeat_penalty": 1.1,
            "num_ctx": 4096
        }
    }
    if json_mode:
        payload["format"] = "json"

    response = requests.post(
        f"{ollama_url}/api/chat",
        json=payload,
        timeout=300
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def build_prompts(question_text, bloom_level, context_text, use_sys, is_strict):
    system = SYSTEM_PROMPT if use_sys else "Jawab pertanyaan pengguna dengan akurat."
    if is_strict and context_text.strip():
        system += STRICT_GROUNDING_ADDON

    user_prompt = f"Tingkat kognitif: {bloom_level}\n{BLOOM[bloom_level]}\n\n"
    if context_text.strip():
        label = "Referensi Wajib" if is_strict else "Konteks Tambahan"
        user_prompt += f"=== {label} ===\n{context_text.strip()}\n==================\n\n"
    user_prompt += f"Pertanyaan: {question_text}"
    return system, user_prompt


def evaluate(question_text, answer_text, bloom_level, context_text, judge_model):
    ref_section = context_text.strip() if context_text.strip() else "Tidak ada referensi khusus."
    prompt = f"""Tingkat Bloom yang diminta: {bloom_level}
Referensi yang diberikan ke model:
{ref_section}

Pertanyaan: {question_text}
Jawaban model:
{answer_text}
"""
    raw = ollama_chat(
        judge_model,
        [
            {"role": "system", "content": EVAL_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temp=0.0,
        top_p_val=0.8,
        json_mode=True
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"notes": raw, "possible_issue": "JSON evaluator gagal diparse."}


def save_history(record):
    Path("data").mkdir(exist_ok=True)
    with open("data/history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


with tab1:
    question = st.text_area(
        "Pertanyaan",
        "Siapa tokoh utama dalam pengembangan sistem listrik AC dan DC?",
        height=100
    )
    bloom = st.selectbox("Tingkat Taksonomi Bloom", list(BLOOM.keys()), index=0)
    st.caption(BLOOM[bloom])

    col_ctx1, col_ctx2 = st.columns([2, 1])
    with col_ctx1:
        custom_context = st.text_area(
            "Konteks / Referensi teks (opsional atau wajib jika Strict Grounding aktif)",
            placeholder="Tempel kutipan artikel, catatan, atau fakta rujukan di sini...",
            height=110
        )
    with col_ctx2:
        uploaded_file = st.file_uploader(
            "Atau unggah file referensi (.txt / .md)",
            type=["txt", "md"]
        )
        file_text = ""
        if uploaded_file is not None:
            file_text = uploaded_file.read().decode("utf-8", errors="ignore")
            st.success(f"File dimuat ({len(file_text)} karakter)")

    combined_context = "\n\n".join(filter(None, [custom_context, file_text]))

    if strict_grounding and not combined_context.strip():
        st.info("⚠️ Mode Referensi Ketat aktif, tetapi kolom referensi masih kosong. Model akan menggunakan mode standar hingga referensi diisi.")

    run = st.button("Jalankan perbandingan", type="primary", use_container_width=True)

    if run and question.strip():
        system, user_prompt = build_prompts(
            question, bloom, combined_context, use_system, strict_grounding
        )
        outputs = []
        cols = st.columns(2)
        for col, model in zip(cols, [model_a, model_b]):
            with col:
                st.subheader(model)
                start = time.time()
                try:
                    answer = ollama_chat(
                        model,
                        [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_prompt}
                        ],
                        temp=temperature,
                        top_p_val=top_p
                    )
                    elapsed = time.time() - start
                    st.markdown(answer)
                    st.caption(f"Waktu respons: {elapsed:.2f} detik")
                    outputs.append((model, answer, elapsed))
                except Exception as e:
                    st.error(f"Gagal menjalankan {model}: {e}")
                    outputs.append((model, "", 0))

        evaluations = {}
        if show_eval and len(outputs) == 2 and all(x[1] for x in outputs):
            st.divider()
            st.subheader(f"Evaluasi otomatis (Juri: {eval_model})")
            st.warning("Skor evaluator berbasis model; gunakan pemeriksaan manusia/sumber tepercaya untuk validasi fakta.")
            eval_cols = st.columns(2)
            for col, (model, answer, elapsed) in zip(eval_cols, outputs):
                with col:
                    st.markdown(f"**{model}**")
                    try:
                        score = evaluate(question, answer, bloom, combined_context, eval_model)
                        evaluations[model] = score
                        numeric = [
                            score.get(k, 0) for k in [
                                "intent_match",
                                "factual_quality",
                                "directness",
                                "conceptual_precision",
                                "bloom_alignment"
                            ]
                        ]
                        m1, m2 = st.columns(2)
                        if all(isinstance(v, (int, float)) for v in numeric):
                            m1.metric("Rata-rata skor / 5", f"{sum(numeric)/5:.2f}")
                        m2.metric("Risiko Halusinasi", score.get("hallucination_risk", "-"))
                        st.json(score)
                    except Exception as e:
                        st.error(f"Evaluasi gagal: {e}")

        if any(x[1] for x in outputs):
            record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "question": question,
                "bloom": bloom,
                "strict_grounding": strict_grounding,
                "has_context": bool(combined_context.strip()),
                "evaluator": eval_model if show_eval else None,
                "models": [
                    {
                        "model": m,
                        "answer": a,
                        "seconds": round(t, 2),
                        "evaluation": evaluations.get(m)
                    }
                    for m, a, t in outputs
                ]
            }
            save_history(record)
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
        with st.expander(f"{level}: {q}"):
            if st.button(f"Uji {level[:2]} dengan {model_a} vs {model_b}", key=f"btn_{level[:2]}"):
                sys_b, usr_b = build_prompts(q, level, "", use_system, False)
                b_cols = st.columns(2)
                for b_col, m_name in zip(b_cols, [model_a, model_b]):
                    with b_col:
                        st.markdown(f"**{m_name}**")
                        try:
                            ans = ollama_chat(
                                m_name,
                                [{"role": "system", "content": sys_b}, {"role": "user", "content": usr_b}],
                                temp=temperature,
                                top_p_val=top_p
                            )
                            st.markdown(ans)
                        except Exception as err:
                            st.error(f"Error: {err}")

with tab3:
    st.subheader("Panduan Penggunaan & Pencegahan Halusinasi")
    st.markdown("""
    1. **Gunakan Temperature Rendah (`0.0 – 0.2`):** Menjaga model tetap deterministik pada pertanyaan faktual (C1–C4).
    2. **Aktifkan Mode Referensi Ketat (*Strict Grounding*):** Saat menguji model berukuran kecil (`1B`), tempelkan teks rujukan atau unggah file `.txt`/`.md` agar model tidak mengarang fakta.
    3. **Pisahkan Model Evaluator:** Pilih model dengan parameter terbesar (misalnya `qwen3.5:4b` atau lebih besar) sebagai juri untuk menilai output model kecil.
    4. **Periksa Kolom `unsupported_claims`:** Fokus pada daftar klaim yang tidak didukung referensi pada hasil JSON evaluator.
    """)
