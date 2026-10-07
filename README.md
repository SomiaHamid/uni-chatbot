# المساعد الجامعي الذكي | SUST Intelligent Academic Assistant

An Arabic-language RAG chatbot that answers student questions about academic
regulations and procedures at **Sudan University of Science and Technology
(SUST)** — grounded in official regulations, with native support for the
**Sudanese Arabic dialect**.

Graduation project (Bachelor of Information Systems, Honours) — College of
Computer Science and Information Technology, SUST, June 2026.

> 🔗 **Live demo:** https://sust-academic-bot.vercel.app/

---

## Why this project

Students at SUST relied almost entirely on administrative staff and long
official PDFs to answer routine academic questions (registration limits,
GPA calculation, exam postponement rules, dismissal conditions, etc.). This
created communication bottlenecks and delays. This project builds an
always-available assistant that answers these questions accurately, in the
students' own dialect, grounded strictly in official university policy —
reducing hallucination risk and the workload on academic advisors.

## What it does

- Understands questions written in **Modern Standard Arabic or Sudanese
  dialect** (e.g. "شنو"، "داير"، "سمستر") via a custom normalization layer.
- Retrieves the most relevant official regulation or FAQ entry using
  **semantic search** (not keyword matching).
- Generates a clear, grounded answer using a **fine-tuned LLM** — and
  explicitly declines to answer when the knowledge base has no reliable
  information, instead of guessing.
- Ships as a chat web app (React) talking to a FastAPI backend.

## Architecture

```
┌───────────┐    HTTP     ┌────────────┐    embed    ┌───────────────┐
│  Frontend │ ──────────► │  Backend   │ ──────────► │ Embedding      │
│  React 18 │ ◄────────── │  FastAPI   │             │ Semantic-Ar-   │
└───────────┘   answer    └─────┬──────┘             │ Qwen-Embed-0.6B│
                                 │  retrieved chunks   └───────┬───────┘
                                 │                              │
                                 ▼                              ▼
                          ┌─────────────┐              ┌───────────────┐
                          │ Qwen3-8B    │              │  ChromaDB      │
                          │ + LoRA      │◄─────────────┤  Vector Store  │
                          └─────────────┘   context     └───────────────┘
```

**Pipeline:** query → Sudanese-dialect normalization → embedding → top-k
cosine similarity search in ChromaDB (k=3, similarity threshold 1.5) →
prompt construction with retrieved context → generation via Qwen3-8B (LoRA
fine-tuned) → response cleaning & de-duplication → answer.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, Tailwind CSS, React Router DOM v6, Axios |
| Backend | FastAPI, Uvicorn, LangChain |
| Embeddings | `Omartificial-Intelligence-Space/Semantic-Ar-Qwen-Embed-0.6B` |
| Vector DB | ChromaDB (cosine similarity) |
| LLM | Qwen3-8B, fine-tuned with LoRA (PEFT) |
| Training/serving compute | Google Colab (GPU) |
| Deployment | Frontend on Vercel · Backend tunneled via ngrok |

## Repository structure

```
.
├── frontend/               # React 18 chat web app
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── .env.example
├── backend/                 # FastAPI + RAG pipeline (production chatbot server)
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
├── training/                # LoRA fine-tuning pipeline for Qwen3-8B
│   ├── finetune_lora.py
│   ├── merge_lora_weights.py
│   └── requirements.txt
└── docs/
    └── evaluation.md         # methodology, results, and honest limitations
```

## Getting started

### Frontend

```bash
cd frontend
npm install
cp .env.example .env      # set VITE_BACKEND_URL to your backend's public URL
npm run dev               # runs on http://localhost:5173
```

### Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env      # fill in your data paths & ngrok token
python main.py            # runs on http://localhost:8000
```

API:
- `GET /start` → greeting + suggested questions
- `POST /chat` → `{"question": "...", "user_id": "..."}` → `{"answer": "...", "suggestions": [...]}`

### Training (fine-tuning Qwen3-8B with LoRA)

```bash
cd training
pip install -r requirements.txt
python finetune_lora.py --data-path ./data/question_fixed.json
python merge_lora_weights.py --lora-path ./qwen3-lora-output
```

## Evaluation results

| Metric | Score |
|---|---|
| BERTScore F1 (semantic similarity to ground truth) | 0.715 |
| Response Relevance (user survey, 5-pt scale) | 4.5 |
| Perceived Ease of Use (user survey, 5-pt scale) | 4.75 |
| Overall satisfaction (user survey, 5-pt scale) | 3.9 |
| Would use again | 100% of surveyed students |

Full methodology and results breakdown: see [`docs/evaluation.md`](docs/evaluation.md).

## Known limitations & honest notes

- Knowledge base currently covers the Academic Advising Guide and
  Examination Regulations only — faculty-specific policies and facilities
  info are out of scope by design.
- Response latency (~20s) is driven by running the LLM inside a Colab GPU
  runtime; a dedicated inference server would reduce this significantly.
- The fine-tuning dataset (~200 Q&A pairs) is small; the RAG mechanism
  (grounding on retrieved regulation text) is the primary safeguard against
  hallucination, not the fine-tuned weights alone.

## Team

- **Alaa Eltayeb Abbas**, **Braah Mohamed Ahmed**, **Somia Hamid Ibrahim**
- Supervisor: **Dr. Yousra Elhakeem**
- Sudan University of Science and Technology, College of Computer Science
  and Information Technology — June 2026
