# Local LLM Prioritizer

Aplikasi lokal untuk membandingkan Llama 3.2 1B dan Qwen 2.5 7B melalui Ollama, dengan prompt prioritas jawaban dan kerangka Taksonomi Bloom Revisi C1-C6.

## Kebutuhan
- Python 3.10+
- Ollama
- Model `llama3.2:1b` dan `qwen2.5:7b`
- RAM yang cukup untuk menjalankan model 7B (bergantung quantization dan konfigurasi)

## Instalasi
```bash
ollama pull llama3.2:1b
ollama pull qwen2.5:7b

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Buka alamat lokal yang ditampilkan Streamlit, biasanya http://localhost:8501.

## Fitur
- Perbandingan dua model secara berurutan.
- Prompt system untuk menjawab inti terlebih dahulu.
- Pemilihan tingkat Bloom C1-C6.
- Evaluasi otomatis 5 dimensi: intent match, factual quality, directness, conceptual precision, Bloom alignment.
- Riwayat hasil ke `data/history.jsonl`.

## Batasan penting
Evaluator otomatis adalah penilaian awal, bukan ground truth. Model evaluator dapat bias atau salah. Untuk benchmark faktual, siapkan kunci jawaban/rubrik dan verifikasi sumber primer. Tidak ada RAG atau pencarian internet pada versi ini.
