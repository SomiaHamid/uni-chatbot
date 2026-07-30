# -*- coding: utf-8 -*-
"""
SUST Academic Assistant - Backend (RAG + FastAPI)
==================================================

نظام مساعد جامعي ذكي يعتمد على Retrieval-Augmented Generation (RAG) للإجابة
على استفسارات الطلاب الأكاديمية بجامعة السودان للعلوم والتكنولوجيا.

المكونات الأساسية:
  1. نموذج تضمين (Embedding): Semantic-Ar-Qwen-Embed-0.6B
  2. قاعدة بيانات متجهية (Vector DB): ChromaDB
  3. نموذج اللغة (LLM): Qwen3-8B مدرب بتقنية LoRA
  4. واجهة API: FastAPI

ملاحظة: هذا الملف مصمم للتشغيل داخل بيئة Google Colab (GPU) مع تعريض
الخدمة عبر ngrok. يمكن أيضاً تشغيله على أي سيرفر يدعم PyTorch + CUDA
بعد ضبط المسارات في ملف `.env`.
"""

import os
import re
import json
import shutil
import threading
from datetime import datetime
from collections import defaultdict, deque
from functools import lru_cache

import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (كل المسارات والأسرار تُقرأ من متغيرات البيئة، لا شيء مكتوب صريح)
# ---------------------------------------------------------------------------

QA_FILE = os.getenv("QA_FILE_PATH", "./data/question.json")
REG_FILE = os.getenv("REGULATIONS_FILE_PATH", "./data/regulations_final.json")
GUIDE_FILES = [
    p.strip() for p in os.getenv("GUIDE_FILES", "").split(",") if p.strip()
]

DB_PERSIST_PATH = os.getenv("VECTOR_DB_PATH", "./data/University_Bot_DB")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "Omartificial-Intelligence-Space/Semantic-Ar-Qwen-Embed-0.6B"
)
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "BraahMohamed1/Qwen3-8B-MyLORA")

CHAT_HISTORY_LOG = os.getenv("CHAT_HISTORY_LOG_PATH", "./data/chat_history.jsonl")

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 3))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 1.5))
DUPLICATE_OVERLAP_THRESHOLD = 0.85

NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")  # يُضبط في .env فقط، أبداً في الكود
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))


# ---------------------------------------------------------------------------
# 1. Embedding Model
# ---------------------------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# ---------------------------------------------------------------------------
# 2. Data Loading (PDF + JSON)
# ---------------------------------------------------------------------------

def load_academic_guide(pdf_paths):
    """يقرأ لوائح الجامعة الرسمية (PDF) ويقسمها إلى مقاطع نصية (chunks)."""
    all_docs = []
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"⚠️ Warning: PDF not found -> {pdf_path}")
            continue

        print(f" Processing: {pdf_path}")
        reader = PdfReader(pdf_path)
        full_text = "".join(
            page.extract_text() + "\n" for page in reader.pages if page.extract_text()
        ).replace("\n\n", "\n")

        chunk_size, overlap, start = 2000, 200, 0
        while start < len(full_text):
            chunk = full_text[start: start + chunk_size]
            all_docs.append(
                Document(
                    page_content=chunk,
                    metadata={"type": "Guide", "source": os.path.basename(pdf_path)},
                )
            )
            start += chunk_size - overlap

    return all_docs


def load_json_data(file_path, doc_type):
    """يقرأ ملفات الأسئلة والأجوبة (QA) واللوائح المقسمة (Regulations)."""
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: JSON not found -> {file_path}")
        return []

    docs = []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
        for item in data:
            answer = item.get("output", "")
            variants = item.get("instruction_variants", [])
            base = item.get("instruction", "")

            all_questions = variants if variants else ([base] if base else [])

            for question in all_questions:
                if question:
                    docs.append(
                        Document(
                            page_content=f"{question}\n{answer}",
                            metadata={"type": doc_type, "answer": answer},
                        )
                    )

    print(f"  ↳ {doc_type}: loaded {len(docs)} documents from {os.path.basename(file_path)}")
    return docs


def load_all_data():
    docs = []
    docs.extend(load_json_data(QA_FILE, "QA"))
    docs.extend(load_json_data(REG_FILE, "Regulation"))
    docs.extend(load_academic_guide(GUIDE_FILES))
    print(f"✅ Total documents loaded: {len(docs)}")
    return docs


# ---------------------------------------------------------------------------
# 3. Vector Database (ChromaDB)
# ---------------------------------------------------------------------------

def _db_is_valid(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "chroma.sqlite3"))


def get_vectordb(force_rebuild: bool = False):
    """
    force_rebuild=True  -> يمسح قاعدة البيانات القديمة ويبنيها من جديد
    force_rebuild=False -> يحمّل الموجودة، أو يبنيها إذا لم تكن موجودة
    """
    if force_rebuild and _db_is_valid(DB_PERSIST_PATH):
        print("[VectorDB] Deleting old database for rebuild...")
        shutil.rmtree(DB_PERSIST_PATH)

    if _db_is_valid(DB_PERSIST_PATH):
        print("[VectorDB] Existing database found — Loading...")
        return Chroma(persist_directory=DB_PERSIST_PATH, embedding_function=embeddings)

    print("[VectorDB] No database found — Building from scratch...")
    docs = load_all_data()
    if not docs:
        raise ValueError("No documents to build the database!")

    print(f"[VectorDB] Building with {len(docs)} documents...")
    db = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=DB_PERSIST_PATH
    )
    print(f"[VectorDB] ✅ Database built and saved → {DB_PERSIST_PATH}")
    return db


try:
    vectordb = get_vectordb(force_rebuild=False)
except Exception as e:
    print(f"❌ Error initializing DB: {e}")
    vectordb = None


# ---------------------------------------------------------------------------
# 4. Language Model (Qwen3-8B + LoRA)
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.eval()
print("✅ Model loaded")


# ---------------------------------------------------------------------------
# 5. RAG Pipeline - Core Logic
# ---------------------------------------------------------------------------

memory_store = defaultdict(lambda: deque(maxlen=6))

_RE_GREETING = re.compile(r"(السلام عليكم|مرحبا|اهلا|هلا|صباح الخير|مساء الخير)")
_RE_THANKS = re.compile(r"(شكرا|شكراً|مشكور|يسلموا)")
_RE_CASUAL = re.compile(r"(كيفك|كيف حالك|اخبارك)")
_RE_YES = re.compile(r"(ايوة|اي|نعم|ايوا)")


def add_memory(user_id, role, message):
    memory_store[user_id].append((role, message))


def get_memory(user_id):
    return memory_store[user_id]


def classify_message(text):
    """يصنف رسالة المستخدم: تحية / شكر / دردشة عابرة / سؤال فعلي."""
    text_lower = text.lower().strip()

    is_greeting = bool(_RE_GREETING.search(text_lower))
    is_thanks = bool(_RE_THANKS.search(text_lower))
    is_casual = bool(_RE_CASUAL.search(text_lower))
    is_yes = bool(_RE_YES.search(text_lower))

    # رسالة طويلة نسبياً تحتوي كلمة تحية لا تعني بالضرورة أنها مجرد تحية
    if len(text.strip()) > 15 and (is_greeting or is_thanks or is_casual or is_yes):
        return "question"

    if is_greeting:
        return "greeting"
    if is_thanks:
        return "thanks"
    if is_casual:
        return "casual_chat"
    if is_yes:
        return "yes_chat"
    if len(text) < 3:
        return "unclear"
    return "question"


def handle_greeting():
    return "اهلا وسهلا 🌟 كيف أقدر أساعدك؟"


def handle_thanks():
    return "العفو 🙌"


def handle_casual_chat():
    return "الحمد لله تمام 😊 كيف أقدر أساعدك؟"


def handle_yes_chat():
    return "اتفضل 😊 كيف أقدر أساعدك؟"


def handle_unclear():
    return "ممكن توضّح سؤالك أكتر؟"


def _is_duplicate(new_text: str, seen: list, threshold: float = DUPLICATE_OVERLAP_THRESHOLD) -> bool:
    """يتحقق إذا كان مقطع نصي جديد يتشابه كثيراً مع مقطع تم استرجاعه مسبقاً."""
    new_chars = set(new_text)
    for s in seen:
        s_chars = set(s)
        overlap = len(new_chars & s_chars) / max(len(new_chars | s_chars), 1)
        if overlap >= threshold:
            return True
    return False


@lru_cache(maxsize=256)
def retrieve_context(query, k=RETRIEVAL_K):
    """يبحث في قاعدة البيانات المتجهية عن أقرب المقاطع دلالياً للسؤال."""
    try:
        results = vectordb.similarity_search_with_score(query, k=k)
        if not results:
            return ""

        parts, seen_texts = [], []
        for doc, score in results:
            print(f"[Retrieve] score={score:.2f} | {doc.page_content[:60]}")
            if score > SIMILARITY_THRESHOLD:
                continue
            if _is_duplicate(doc.page_content, seen_texts):
                print("[Retrieve] skipped duplicate chunk")
                continue
            seen_texts.append(doc.page_content)
            ans = doc.metadata.get("answer", "")
            parts.append(doc.page_content + ("\n" + ans if ans else ""))

        return "\n\n".join(parts)
    except Exception as e:
        print(f"[Retrieve] Error: {e}")
        return ""


def build_prompt(question, context, memory_deque):
    """يبني الـ prompt النهائي المرسل للنموذج اللغوي مع تعليمات صارمة للالتزام بالسياق."""
    history = "\n".join(f"{r}: {m}" for r, m in list(memory_deque)[-6:])
    history_block = f"\nسجل المحادثة:\n{history}\n" if history else ""

    return f"""أنت مساعد جامعي متخصص. مهمتك الإجابة على أسئلة الطلاب بدقة تامة.

التعليمات الصارمة:
1. أجب فقط من المعلومات الموجودة في السياق أدناه، لا تضف أي معلومة من عندك
2. اكتب الأسماء والمصطلحات كما هي في السياق بالضبط (مثلاً: "الوحدة الطبية"، "عميد الكلية")
3. أعد الصياغة بأسلوب واضح، لا تنسخ حرفياً إلا للتعاريف الرسمية
4. اكتب الإجابة مرة واحدة فقط، لا تكررها
5. إذا المعلومات غير كافية قل فقط: عذرا، لا تتوفر لدي معلومات مؤكدة حول هذا الاستفسار. يُرجى مراجعة المرشد الأكاديمي أو القسم المختص للحصول على الإجابة الصحيحة
6. أجب على السؤال المطروح فقط دون إضافة أي إجابات أخرى أو إعادة كتابة سؤال آخر
7. أجب باللغة العربية الفصحى مع الالتزام التام بقواعد الإملاء والنحو وتجنب الأخطاء تماماً
{history_block}
السياق:
{context}

السؤال:
{question}

الإجابة:
"""


_RE_CUT = re.compile(
    r"(---|السؤال:|سؤال:|المستخدم:|User:|Human:|الإجابة[\s\(:])", re.IGNORECASE
)


def clean_response(text: str) -> str:
    """ينظف مخرجات النموذج من التكرار والاقتطاعات غير المرغوبة."""
    m = _RE_CUT.search(text)
    if m:
        text = text[: m.start()]

    text = text.strip().lstrip("-").strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    seen, out = [], []
    for line in lines:
        key = line[:80]
        is_near_dup = any(
            len(set(key) & set(s)) / max(len(set(key) | set(s)), 1) > 0.85 for s in seen
        )
        if not is_near_dup:
            seen.append(key)
            out.append(line)
    return "\n".join(out).strip()


def save_chat_interaction(question: str, retrieved_context, answer: str):
    """يسجّل كل تفاعل في ملف JSONL لأغراض التقييم والتحسين المستقبلي."""
    os.makedirs(os.path.dirname(CHAT_HISTORY_LOG) or ".", exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "retrieved_context": retrieved_context,
        "answer": answer,
    }
    with open(CHAT_HISTORY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_llm(prompt):
    """يستدعي نموذج Qwen3-8B لتوليد الإجابة، مع معالجة نفاد ذاكرة الـ GPU."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(
        model.device
    )
    input_length = inputs["input_ids"].shape[1]

    generate_kwargs = dict(
        max_new_tokens=400,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=6,
    )

    try:
        with torch.no_grad():
            out = model.generate(**inputs, **generate_kwargs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        with torch.no_grad():
            out = model.generate(**inputs, **generate_kwargs)

    new_tokens = out[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


_answer_cache: dict = {}


def rag_chat(question, user_id="default_user"):
    """نقطة الدخول الرئيسية لمعالجة سؤال المستخدم عبر خط أنابيب RAG الكامل."""
    msg_type = classify_message(question)

    if msg_type == "greeting":
        return handle_greeting()
    if msg_type == "thanks":
        return handle_thanks()
    if msg_type == "casual_chat":
        return handle_casual_chat()
    if msg_type == "yes_chat":
        return handle_yes_chat()
    if msg_type == "unclear":
        return handle_unclear()

    context = retrieve_context(question)
    if not context or len(context.strip()) < 50:
        return (
            "عذرا، لا تتوفر لدي معلومات مؤكدة حول هذا الاستفسار. "
            "يُرجى مراجعة المرشد الأكاديمي أو القسم المختص للحصول على الإجابة الصحيحة"
        )

    cache_key = f"{user_id}||{question}"
    if cache_key in _answer_cache:
        return _answer_cache[cache_key] + "\nهل عندك أي استفسار آخر؟ 😊"

    memory = get_memory(user_id)
    prompt = build_prompt(question, context, memory)
    answer = clean_response(call_llm(prompt))

    save_chat_interaction(question=question, retrieved_context=context, answer=answer)

    _answer_cache[cache_key] = answer
    add_memory(user_id, "user", question)
    add_memory(user_id, "assistant", answer)

    return answer + "\nهل عندك أي استفسار آخر؟ 😊"


# ---------------------------------------------------------------------------
# 6. FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(title="SUST Academic Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يُفضّل تقييدها لدومين الفرونت-إند الفعلي في الإنتاج
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str
    user_id: str = "default_user"


SUGGESTIONS = [
    "كيف يتم حساب المعدل الفصلي؟",
    "كيف أجمد دراستي؟",
    "ما هو نظام الساعات المعتمدة؟",
    "ما هي شروط تأجيل الجلوس للامتحان؟",
]


@app.get("/start")
def start():
    return {"message": "كيف أقدر أساعدك؟", "suggestions": SUGGESTIONS}


@app.post("/chat")
def chat(data: Question):
    try:
        answer = rag_chat(data.question, data.user_id)
        return {"answer": answer, "suggestions": SUGGESTIONS}
    except Exception as e:
        print(f"[/chat] Error: {e}")
        return {"answer": "حصل خطأ، حاول مرة أخرى", "suggestions": SUGGESTIONS}


# ---------------------------------------------------------------------------
# 7. Server Entry Point
# ---------------------------------------------------------------------------

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)


def start_with_ngrok():
    """يشغل السيرفر محلياً ويعرضه عبر ngrok (للاستخدام داخل Google Colab)."""
    from pyngrok import ngrok

    if not NGROK_AUTH_TOKEN:
        raise RuntimeError(
            "NGROK_AUTH_TOKEN غير موجود في متغيرات البيئة. أضفه إلى ملف .env "
            "(انظر .env.example)."
        )

    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    public_url = ngrok.connect(SERVER_PORT)
    print(f"🌐 Public URL: {public_url}")
    return public_url


if __name__ == "__main__":
    # للتشغيل المحلي العادي (سيرفر حقيقي وليس Colab):
    #   uvicorn main:app --host 0.0.0.0 --port 8000
    # للتشغيل داخل Google Colab مع ngrok:
    #   from main import start_with_ngrok; start_with_ngrok()
    run_server()
